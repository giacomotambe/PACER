"""NMPC uniciclo v2 (rev. 2026-11-17, nota di revisione §3r). Stessa idea di
unicycle_nmpc.py (vincoli rigidi sulle regioni calibrate), con quattro correzioni:

1. Limiti di accelerazione: v_k - v_{k-1} in [-A_DEC*dt, A_ACC*dt], |w_k - w_{k-1}| <= ALPHA*dt,
   a partire dalla velocita' misurata (v_mis, w_mis).
2. Vincoli fra i campioni: in ogni intervallo [t_k, t_{k+1}] (k = 0..N-1, con p_0 = posizione
   attuale e mu_0 = posizione attuale della persona) il vincolo vale in s = 1/3, 2/3, 1 con moto
   relativo lineare e la regione dell'offset k+1. Copre anche il primo tratto, quello che si
   esegue davvero. Fra due punti consecutivi (al massimo v_max*dt/3 = 0.21 m) l'intrusione
   possibile in una regione di raggio R e' R - sqrt(R^2 - 0.107^2) < 0.6 cm per R >= 1 m.
   Il punto s = 0 del primo intervallo (posizione attuale, non controllabile) e' escluso:
   un vincolo su di esso renderebbe il problema infattibile appena una persona e' vicina.
   I bordi (marciapiede) sono semipiani n.p >= d, non dischi.
   Passive safety: negli intervalli in cui il robot e' fermo (v_k < EPS_V) il vincolo verso le
   persone e' rilasciato linearmente: una persona che va addosso a un robot fermo non e' una
   collisione attiva. Senza questo rilascio l'arresto terminale e' infattibile davanti a un
   flusso contrario (marciapiede): il robot fermo cadrebbe nelle regioni future di chi arriva.
3. Vincolo terminale di arresto v_{N-1} = 0: ogni piano finisce con il robot fermo, e la
   frenata rispetta le regioni. Il piano traslato piu' uno stop in coda e' un candidato
   fattibile al ciclo dopo, salvo persone la cui nuova regione cade sul punto di arresto
   (fattibilita' ricorsiva nel senso della passive safety).
4. CasADi + IPOPT con derivate esatte e multi-start "lazy" (piano precedente traslato; solo se
   non e' fattibile: rettilineo, curva a sinistra, curva a destra), al massimo MAX_ITER = 50
   iterazioni per avvio. Velocita' (rev. 2026-11-18, nota §3s): potatura esatta dei vincoli
   irraggiungibili (_prune), solutori per taglie a gradini fini, nessun nuovo avvio dopo
   T_BUDGET = 0.25 s. Ridurre le iterazioni a 30 o gli avvii a 1-2 e' piu' veloce ma nel
   marciapiede affollato allunga il tempo al goal del 10-25% (run v20-v23).
   Ottimalita' locale; la sicurezza e' garantita dal controllo esplicito e dal ripiego.

Costo sociale opzionale (punto 5, PACER_SOCIAL): vedi social_terms().
Ostacoli statici (costmap Nav2, ROS2): costo soft relu(d_safe - d)^2 su tutto l'orizzonte,
fino a S_MAX punti, come in unicycle_nmpc.py. Parametri modificabili con configure() (nodo
ROS2) o con variabili d'ambiente (benchmark).

Interfaccia: solve_step(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, u_warm)
-> (u0, ok, u_full, margin, status), status in {optimal, fallback_stop, fallback_evasive,
infeasible}. obstacles: lista di dict {mu0: (2,), mu: (N,2), Sigma_inv: (N,2,2), q2: (N,)}
(q2 = 0 disattiva il vincolo a quell'offset).
"""
import os
import threading
import time

import numpy as np
import casadi as ca

DT = 0.4
A_ACC = float(os.environ.get("PACER_A_ACC", "1.0"))     # [m/s^2]
A_DEC = float(os.environ.get("PACER_A_DEC", "1.5"))     # [m/s^2], uguale a ESTOP_A
ALPHA = float(os.environ.get("PACER_ALPHA_W", "3.0"))   # [rad/s^2]
TERMINAL_STOP = os.environ.get("PACER_TERMINAL", "1") == "1"
# multi-start: "lazy" = alternative solo se la prima soluzione non e' fattibile; "full" = sempre
MULTISTART = os.environ.get("PACER_MULTISTART", "lazy")
WEIGHTS = (6.0, 0.15, 0.02, 0.05)   # goal finale, goal lungo il piano, |u|^2, |du|^2 (come v1)
BUCKETS = (0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64)   # ostacoli: solutori compilati per taglia fissa
SUBS = (1.0 / 3.0, 2.0 / 3.0, 1.0)   # punti di controllo in ogni intervallo
EPS_V = 0.05                         # [m/s] sotto questa velocita' il robot e' "fermo"
H_MAX = 4                            # semipiani (bordi)
S_MAX = 32                           # ostacoli statici (costo soft)
H_BUCKETS = (0, 2, 4)
S_BUCKETS = (0, 8, 16, 32)
# Velocita' (rev. 2026-11-18): potatura esatta delle regioni irraggiungibili, taglie a gradini
# fini, tempo massimo per il solutore (oltre: ripiego). Vedi _prune() e solve_step().
PRUNE = os.environ.get("PACER_PRUNE", "1") == "1"
PRUNE_MARGIN = 0.10                  # [m] margine della potatura
T_BUDGET = float(os.environ.get("PACER_T_BUDGET", "0.25"))   # [s] dopo questo tempo niente nuovi avvii, 0 = nessun limite
MAX_ITER = int(os.environ.get("PACER_MAX_ITER", "50"))   # iterazioni IPOPT, oltre: ripiego
IPOPT_EXTRA = {}                     # opzioni IPOPT aggiuntive (profilazione)
# JIT (rev. 2026-11-18, nota §3s): le funzioni del problema (costo, vincoli, gradienti, jacobiano,
# hessiana) generate in C e compilate con un compilatore di sistema (gcc sul robot). Cache su disco
# indicizzata dall'hash del codice: si compila una volta per taglia, poi si ricarica. Se la
# compilazione fallisce si usa il solutore interpretato (stesso problema, piu' lento).
JIT = os.environ.get("PACER_JIT", "0") == "1"
JIT_CC = os.environ.get("PACER_JIT_CC", "gcc")
JIT_FLAGS = os.environ.get("PACER_JIT_FLAGS", "-O1")
JIT_DIR = os.environ.get("PACER_JIT_DIR", os.path.join(os.path.expanduser("~"), ".cache", "pacer_mpc_jit"))
JIT_TIMEOUT = 900                    # [s] per una compilazione
_JIT_FAILED = []                     # messaggi di errore (il primo e' stampato una volta)
# Ibrido (rev. 2026-11-19, nota §3s): prima un SQP con QP ad insieme attivo (qpOASES) dal primo
# avvio; se il piano non e' fattibile, IPOPT con il multi-start di sempre. Offline: 9 ms invece di
# 32 quando SQP riesce (75% dei problemi), stesso tasso di successo complessivo.
HYBRID = os.environ.get("PACER_HYBRID", "0") == "1"
# Vincoli a penalita' esatta (rev. 2026-11-19, nota §3s): i vincoli verso persone e bordi
# dell'intervallo k diventano g >= -s_k con s_k >= 0 e costo W_SLACK * sum(s_k). Il problema ha
# sempre soluzione e IPOPT converge anche quando il problema vero non ne ha (prima esauriva le
# iterazioni: 62% del tempo). Il piano si accetta solo se s = 0 entro la tolleranza (controllo
# esplicito sui vincoli veri, come prima); altrimenti si passa al ripiego. Con W_SLACK maggiore dei
# moltiplicatori la penalita' e' esatta: se il problema vero ha soluzione, s = 0 all'ottimo.
SOFT = os.environ.get("PACER_SOFT", "1") == "1"
W_HEAD = float(os.environ.get("PACER_W_HEAD", "0.3"))   # costo di orientamento verso il goal (orizzonti corti)
W_SLACK = float(os.environ.get("PACER_W_SLACK", "1000.0"))
_KIDX = {}                           # id(solver) -> intervallo di ogni vincolo persone/bordi
SOFT_EARLY = os.environ.get("PACER_SOFT_EARLY", "1") == "1"   # niente altri avvii se lo scarto e' netto
SOFT_INFEAS = 1e-2                   # scarto oltre il quale il problema e' dichiarato irrisolvibile
SQP_MAX_ITER = 50
_SQP = {}
W_STATIC = float(os.environ.get("PACER_W_STATIC", "4000.0"))

_CACHE = {}
_DUMP = os.environ.get("PACER_DUMP")   # file pickle dei problemi (profilazione offline)


def set_dt(dt):
    global DT
    DT = dt
    _CACHE.clear()


def configure(**kw):
    """Imposta i parametri del modulo (nomi come le costanti: A_ACC, A_DEC, ALPHA,
    TERMINAL_STOP, MULTISTART, WEIGHTS, SOCIAL, W_COMFORT, D_COMFORT, W_SIDE, W_FRONT,
    W_PATH, W_STATIC, EPS_V, PRUNE, MAX_ITER, T_BUDGET, JIT, JIT_CC, JIT_FLAGS, JIT_DIR) e svuota la
    cache dei solutori."""
    g = globals()
    for k, v in kw.items():
        if k not in g:
            raise KeyError(f"parametro sconosciuto: {k}")
        g[k] = v
    _CACHE.clear()


def _build(N, J, social, Hn=H_MAX, Sn=S_MAX):
    """Solutore IPOPT per N passi e J ostacoli (taglia fissa, slot vuoti con q2 = 0)."""
    U = ca.SX.sym("U", 2 * N)
    kidx = []                        # intervallo di ogni vincolo dopo quelli di accelerazione
    npar = 3 + 2 + 2 + 4 + J * (2 + N * 2 + N * 4 + N) + 3 * Hn + 3 * Sn + (1 if social else 0)
    P = ca.SX.sym("P", npar)
    x0, um, goal = P[0:3], P[3:5], P[5:7]
    path_dir, path_org = P[7:9], P[9:11]   # retta del percorso globale (costo sociale, punto 5)
    off = 11
    px, py, th = x0[0], x0[1], x0[2]
    pos = [ca.vertcat(px, py)]
    ths = []
    for k in range(N):
        v, w = U[2 * k], U[2 * k + 1]
        px = px + v * ca.cos(th) * DT
        py = py + v * ca.sin(th) * DT
        th = th + w * DT
        pos.append(ca.vertcat(px, py))
        ths.append(th)
    w_goal, w_stage, w_u, w_du = WEIGHTS
    J_cost = w_goal * ca.sumsqr(pos[-1] - goal)
    for k in range(1, N + 1):
        J_cost += w_stage * ca.sumsqr(pos[k] - goal)
    J_cost += w_u * ca.sumsqr(U)
    for k in range(1, N):
        J_cost += w_du * ca.sumsqr(U[2 * k:2 * k + 2] - U[2 * k - 2:2 * k])
    if W_HEAD > 0:
        # orientamento verso il goal (rev. 2026-11-19): con orizzonte corto e goal alle spalle,
        # girarsi sul posto non riduce il costo di posizione entro l'orizzonte e restare fermi e' un
        # minimo locale; questo termine lo rimuove. Pesato per |d|/sqrt(|d|^2 + 0.25): ~0 sul goal.
        dg = goal - x0[0:2]
        nd = ca.sqrt(ca.sumsqr(dg) + 1e-9)
        fade = nd / ca.sqrt(nd ** 2 + 0.25)
        for th_k in ths:
            J_cost += W_HEAD * fade * (1 - (ca.cos(th_k) * dg[0] + ca.sin(th_k) * dg[1]) / nd)

    g, lbg, ubg = [], [], []
    # 1) limiti di accelerazione (lineari)
    prev_v, prev_w = um[0], um[1]
    for k in range(N):
        g += [U[2 * k] - prev_v, U[2 * k + 1] - prev_w]
        lbg += [-A_DEC * DT, -ALPHA * DT]
        ubg += [A_ACC * DT, ALPHA * DT]
        prev_v, prev_w = U[2 * k], U[2 * k + 1]
    n_rate = len(g)
    # 2) vincoli sui segmenti, uno per ostacolo e intervallo
    soc_obs = []
    for j in range(J):
        mu0 = P[off:off + 2]; off += 2
        mu = ca.reshape(P[off:off + 2 * N], 2, N); off += 2 * N
        Si = P[off:off + 4 * N]; off += 4 * N
        q2 = P[off:off + N]; off += N
        prev_mu = mu0
        for k in range(N):
            S = ca.reshape(Si[4 * k:4 * k + 4], 2, 2)
            a = pos[k] - prev_mu
            b = (pos[k + 1] - mu[:, k]) - a
            moving = 1 - ca.fmax(0, 1 - U[2 * k] / EPS_V)     # 0 da fermo, 1 sopra EPS_V
            for sf in SUBS:
                r = a + sf * b
                g.append(ca.mtimes([r.T, S, r]) - q2[k] * moving)
                lbg.append(0.0); ubg.append(ca.inf); kidx.append(k)
            prev_mu = mu[:, k]
        soc_obs.append((mu, Si, q2))
    for h in range(Hn):
        nx, ny, d = P[off], P[off + 1], P[off + 2]; off += 3
        for k in range(N):
            for sf in SUBS:
                p = pos[k] + sf * (pos[k + 1] - pos[k])
                g.append(nx * p[0] + ny * p[1] - d)
                lbg.append(0.0); ubg.append(ca.inf); kidx.append(k)
    for s_ in range(Sn):
        sx, sy, ds = P[off], P[off + 1], P[off + 2]; off += 3
        for k in range(1, N + 1):
            dd = ca.sqrt((pos[k][0] - sx) ** 2 + (pos[k][1] - sy) ** 2 + 1e-9)
            J_cost += W_STATIC * ca.fmax(0, ds - dd) ** 2
    if social:
        J_cost += social_terms(pos, U, soc_obs, path_dir, path_org, P[off], N)
    if SOFT:
        Sl = ca.SX.sym("S", N)
        g_nlp = g[:n_rate] + [gi + Sl[k] for gi, k in zip(g[n_rate:], kidx)]
        nlp = {"x": ca.vertcat(U, Sl), "p": P, "f": J_cost + W_SLACK * ca.sum1(Sl), "g": ca.vertcat(*g_nlp)}
    else:
        nlp = {"x": U, "p": P, "f": J_cost, "g": ca.vertcat(*g)}
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": MAX_ITER, "ipopt.tol": 1e-5,
            "ipopt.acceptable_tol": 1e-3, "ipopt.acceptable_iter": 5, "ipopt.sb": "yes",
            "error_on_fail": False}
    opts.update({"ipopt." + k: v for k, v in IPOPT_EXTRA.items()})
    solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
    if JIT:
        solver = _jit(solver, opts) or solver
    if HYBRID:
        _SQP[id(solver)] = ca.nlpsol("mpc_sqp", "sqpmethod", nlp, {
            "qpsol": "qpoases", "qpsol_options": {"printLevel": "none", "error_on_fail": False},
            "convexify_strategy": "regularize", "max_iter": SQP_MAX_ITER, "print_header": False,
            "print_iteration": False, "print_status": False, "print_time": False, "error_on_fail": False})
    gfun = ca.Function("g", [U, P], [ca.vertcat(*g)])     # vincoli veri, senza scarti
    _KIDX[id(solver)] = np.array(kidx, int)
    return solver, gfun, np.array(lbg), np.array(ubg), n_rate


# --- costo sociale (punto 5) -------------------------------------------------------------
W_COMFORT = float(os.environ.get("PACER_W_COMFORT", "2.0"))
D_COMFORT = float(os.environ.get("PACER_D_COMFORT", "1.2"))   # [m] distanza sociale desiderata
W_SIDE = float(os.environ.get("PACER_W_SIDE", "0.5"))
W_FRONT = float(os.environ.get("PACER_W_FRONT", "1.0"))
W_PATH = float(os.environ.get("PACER_W_PATH", "0.5"))


def social_terms(pos, U, obs, path_dir, path_org, on, N):
    """Termini soft, moltiplicati per il flag `on` (parametro, 0 o 1):
    - comfort: penalita' quadratica se la distanza dal centro predetto e' sotto D_COMFORT;
    - lato: circolazione a destra. Incrociando o sorpassando, chi tiene la destra lascia la
      persona alla propria sinistra, cioe' il robot sta alla SINISTRA della persona rispetto
      alla direzione di lei: si penalizza il robot alla destra della persona (prodotto
      vettoriale < 0), solo entro circa 3 m;
    - fronte: penalita' per stare davanti alla persona entro 2 m nella sua direzione di moto;
    - percorso: penalita' sullo scostamento laterale dalla retta del percorso globale
      (path_org, path_dir)."""
    c = 0
    for mu, Si, q2 in obs:
        active = ca.if_else(q2[0] > 0, 1, 0)       # slot vuoti e vincoli spenti non contano
        for k in range(N):
            r = pos[k + 1] - mu[:, k]
            d2 = ca.sumsqr(r)
            c += W_COMFORT * active * ca.fmax(0, D_COMFORT ** 2 - d2) ** 2
            dk = mu[:, k] - (mu[:, k - 1] if k > 0 else mu[:, k])
            if k == 0:
                dk = mu[:, 1] - mu[:, 0]
            nd = dk / ca.sqrt(ca.sumsqr(dk) + 1e-6)
            cross = nd[0] * r[1] - nd[1] * r[0]          # > 0: robot alla sinistra della persona
            near = ca.exp(-d2 / 9.0)
            c += W_SIDE * active * near * ca.fmax(0, -cross) ** 2
            along = nd[0] * r[0] + nd[1] * r[1]          # > 0: robot davanti alla persona
            c += W_FRONT * active * ca.exp(-d2 / 4.0) * ca.fmax(0, along) ** 2
    # scostamento laterale dalla retta del percorso globale
    for k in range(1, N + 1):
        r = pos[k] - path_org
        lat = path_dir[0] * r[1] - path_dir[1] * r[0]
        c += W_PATH * lat ** 2
    return on * c


SOCIAL = os.environ.get("PACER_SOCIAL", "0") == "1"


def _jit(solver, opts):
    """Genera il C delle funzioni del problema, lo compila (se non e' gia' in cache) e restituisce
    il solutore che usa la libreria compilata. None se qualcosa fallisce."""
    import hashlib
    import shutil
    import subprocess
    import sys
    import tempfile
    ext = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
    tmpd = None
    try:
        os.makedirs(JIT_DIR, exist_ok=True)
        tmpd = tempfile.mkdtemp(dir=JIT_DIR)
        cwd = os.getcwd()
        try:
            os.chdir(tmpd)
            solver.generate_dependencies("nlp.c")        # scrive nella cartella corrente
        finally:
            os.chdir(cwd)
        src = os.path.join(tmpd, "nlp.c")
        with open(src, "rb") as fh:
            h = hashlib.sha1(fh.read() + (JIT_CC + JIT_FLAGS).encode()).hexdigest()[:16]
        lib = os.path.join(JIT_DIR, f"pacer_mpc_{h}{ext}")
        if not os.path.exists(lib):
            t0 = time.perf_counter()
            out = os.path.join(tmpd, "nlp" + ext)
            cmd = [JIT_CC, "-fPIC", "-shared", *JIT_FLAGS.split(), src, "-o", out, "-lm"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=JIT_TIMEOUT)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-2000:])
            os.replace(out, lib)                          # atomico: niente librerie a meta'
            print(f"[unicycle_nmpc_v2] JIT: compilato {os.path.basename(lib)} in "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
        return ca.nlpsol("mpc", "ipopt", lib, opts)
    except Exception as e:                                # noqa: BLE001 (ripiego sull'interpretato)
        if not _JIT_FAILED:
            print(f"[unicycle_nmpc_v2] JIT non disponibile, uso il solutore interpretato: {e}",
                  flush=True)
        _JIT_FAILED.append(str(e))
        return None
    finally:
        if tmpd is not None:
            shutil.rmtree(tmpd, ignore_errors=True)


def precompile(N, buckets=None, halfplanes=(H_MAX,), statics=None):
    """Costruisce (e con JIT compila) i solutori di tutte le taglie di ostacoli, per non
    costruirli durante la navigazione (un solutore nuovo costa da 0.1 a 2 s). halfplanes/statics:
    numeri di semipiani e di ostacoli statici da coprire (default: tutte le taglie di statici).
    Con JIT semipiani e statici restano alla taglia massima."""
    t0 = time.perf_counter()
    for J in (buckets or BUCKETS):
        for H in halfplanes:
            for S in (statics if statics is not None else S_BUCKETS):
                _get(N, J, H, S)
    return time.perf_counter() - t0


def _bucket(n, buckets):
    return next((b for b in buckets if b >= n), n)


def _get(N, J, H=H_MAX, S=S_MAX):
    Jb, Hb, Sb = _bucket(J, BUCKETS), _bucket(min(H, H_MAX), H_BUCKETS), _bucket(min(S, S_MAX), S_BUCKETS)
    if JIT:          # meno librerie da compilare: semipiani e statici sempre alla taglia massima
        Hb, Sb = H_MAX, S_MAX
    key = (N, Jb, Hb, Sb, SOCIAL, DT, JIT, HYBRID, SOFT, W_SLACK, W_HEAD, tuple(sorted(IPOPT_EXTRA.items())))
    if key not in _CACHE:
        _CACHE[key] = _build(N, Jb, SOCIAL, Hb, Sb)
    return _CACHE[key], Jb, Hb, Sb


def _reach(v_mis, v_max, N):
    """R[k]: distanza massima percorribile entro t_k (k = 0..N) partendo da v_mis con
    accelerazione A_ACC e velocita' costante a tratti, come nel modello."""
    R, v, d = [0.0], max(0.0, v_mis), 0.0
    for _ in range(N):
        v = min(v_max, v + A_ACC * DT)
        d += v * DT
        R.append(d)
    return np.array(R)


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _prune(x0, v_mis, v_max, obstacles, N):
    """Potatura esatta: il vincolo dell'ostacolo j nell'intervallo k (punti fra p_k e p_{k+1},
    persona fra mu_{k-1} e mu_k) e' sempre soddisfatto se nessuna posizione raggiungibile
    (disco di raggio R_{k+1} attorno al robot) puo' entrare nella regione, la cui estensione
    massima e' sqrt(q2 / lambda_min(S)). In quel caso q2_k = 0 (vincolo spento); gli ostacoli
    con tutti i vincoli spenti escono dal problema. L'insieme ammissibile non cambia."""
    p = np.asarray(x0[:2], float)
    R = _reach(v_mis, v_max, N)
    out = []
    for o in obstacles:
        q2 = np.array(o["q2"][:N], float)
        mu = np.asarray(o["mu"][:N], float)
        Si = np.asarray(o["Sigma_inv"][:N], float)
        prev = np.asarray(o["mu0"], float)
        for k in range(N):
            if q2[k] > 0 and np.isfinite(q2[k]):
                lam = float(np.linalg.eigvalsh(0.5 * (Si[k] + Si[k].T))[0])
                if lam > 0 and _seg_dist(p, prev, mu[k]) - R[k + 1] > np.sqrt(q2[k] / lam) + PRUNE_MARGIN:
                    q2[k] = 0.0
            prev = mu[k]
        if np.any(q2 > 0):
            o2 = dict(o)
            o2["q2"] = q2
            out.append(o2)
    return out


def _params(x0, um, goal, obstacles, N, Jb, path_dir, path_org, halfplanes=(), statics=(), Hb=H_MAX,
            Sb=S_MAX):
    parts = [np.asarray(x0, float), np.asarray(um, float), np.asarray(goal, float),
             np.asarray(path_dir, float), np.asarray(path_org, float)]
    for o in obstacles:
        # mu (N,2) riga per riga = colonne della reshape column-major di CasADi
        parts += [np.asarray(o["mu0"], float), np.asarray(o["mu"][:N], float).ravel(),
                  np.asarray(o["Sigma_inv"][:N], float).transpose(0, 2, 1).reshape(-1),
                  np.asarray(o["q2"][:N], float)]
    far = np.array([1e4, 1e4])
    for _ in range(Jb - len(obstacles)):
        parts += [far, np.tile(far, N), np.tile(np.eye(2).ravel(), N), np.zeros(N)]
    hp = list(halfplanes)[:Hb]
    for nx, ny, d in hp:
        parts.append(np.array([nx, ny, d], float))
    for _ in range(Hb - len(hp)):
        parts.append(np.array([0.0, 0.0, -1.0]))      # 0 >= -1: sempre soddisfatto
    st = list(statics)
    if len(st) > Sb:          # tiene i piu' vicini al robot
        st.sort(key=lambda o: (o[0] - x0[0]) ** 2 + (o[1] - x0[1]) ** 2)
        st = st[:Sb]
    for sx, sy, ds in st:
        parts.append(np.array([sx, sy, ds], float))
    for _ in range(Sb - len(st)):
        parts.append(np.array([1e4, 1e4, 0.0]))       # d_safe = 0: nessun costo
    if SOCIAL:
        parts.append(np.array([1.0]))
    return np.concatenate(parts)


def rollout(x0, u):
    px, py, th = x0
    out = []
    for v, w in u:
        px += v * np.cos(th) * DT; py += v * np.sin(th) * DT; th += w * DT
        out.append((px, py))
    return np.array(out)


def _inits(x0, um, goal, N, v_max, omega_max, u_warm):
    """Inizializzazioni del multi-start, tutte compatibili con i limiti di accelerazione."""
    v0, w0 = um
    cands = []
    if u_warm is not None and len(u_warm) == N:
        cands.append(np.array(u_warm, float))
    head = np.arctan2(goal[1] - x0[1], goal[0] - x0[0]) - x0[2]
    head = (head + np.pi) % (2 * np.pi) - np.pi
    turns = [float(np.clip(head / (N * DT), -omega_max, omega_max))]
    if MULTISTART != "0":
        turns += [0.8 * omega_max, -0.8 * omega_max]
    for wt in turns:
        u = np.zeros((N, 2)); v, w = v0, w0
        for k in range(N):
            # accelera verso 0.7 v_max, poi frena per l'arresto terminale
            k_left = N - 1 - k
            v_cap = A_DEC * DT * k_left if TERMINAL_STOP else v_max
            v = float(np.clip(min(0.7 * v_max, v_cap), v - A_DEC * DT, v + A_ACC * DT))
            v = max(0.0, min(v, v_max))
            w = float(np.clip(wt, w - ALPHA * DT, w + ALPHA * DT))
            u[k] = (v, w)
        cands.append(u)
    if MULTISTART == "0":
        cands = cands[:1]
    elif MULTISTART == "one":          # piano precedente + rettilineo verso il goal
        cands = cands[:2]
    return cands


def solve_step(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, u_warm=None,
               feas_tol=1e-4, path_dir=None, path_org=None, halfplanes=(), statics=()):
    """statics: lista di (x, y, d_safe) per il costo soft degli ostacoli statici.
    path_org/path_dir: retta del percorso globale (default: dal robot verso il goal)."""
    if _DUMP:
        import pickle
        with open(_DUMP, "ab") as fh:
            pickle.dump(dict(x0=np.asarray(x0, float), v_mis=v_mis, w_mis=w_mis, goal=np.asarray(goal, float),
                             obstacles=obstacles, N=N, v_max=v_max, omega_max=omega_max, u_warm=u_warm,
                             path_dir=path_dir, path_org=path_org, halfplanes=list(halfplanes),
                             statics=list(statics)), fh)
    t0 = time.perf_counter()
    goal = np.asarray(goal, float)
    if path_dir is None:
        d = goal - np.asarray(x0[:2], float)
        path_dir = d / (np.linalg.norm(d) + 1e-9)
    if path_org is None:
        path_org = np.asarray(x0[:2], float)
    if PRUNE:
        obstacles = _prune(x0, v_mis, v_max, obstacles, N)
    halfplanes, statics = list(halfplanes)[:H_MAX], list(statics)
    (solver, gfun, lbg, ubg, n_rate), Jb, Hb, Sb = _get(N, len(obstacles), len(halfplanes), len(statics))
    P = _params(x0, (v_mis, w_mis), goal, obstacles, N, Jb, path_dir, path_org, halfplanes, statics, Hb, Sb)
    lbx = np.tile([0.0, -omega_max], N).astype(float)
    ubx = np.tile([v_max, omega_max], N).astype(float)
    if TERMINAL_STOP:
        ubx[2 * (N - 1)] = 0.0
    best = None
    inits = _inits(x0, (v_mis, w_mis), goal, N, v_max, omega_max, u_warm)
    attempts = [(i, u, solver) for i, u in enumerate(inits)]
    if HYBRID and id(solver) in _SQP:
        attempts = [(0, inits[0], _SQP[id(solver)])] + attempts
    solve_step.last_solver = None
    for i_init, u_init, slv in attempts:
        if MULTISTART in ("lazy", "one", "0") and best is not None:
            break
        if T_BUDGET > 0 and i_init >= 1 and time.perf_counter() - t0 > T_BUDGET:
            break
        x_init = np.clip(u_init.ravel(), lbx, ubx)
        lbx_s, ubx_s = lbx, ubx
        if SOFT:     # scarti iniziali: quanto serve perche' il punto di partenza sia ammissibile
            kidx = _KIDX[id(solver)]
            gv0 = np.array(gfun(x_init, P)).ravel()[n_rate:]
            s0 = np.zeros(N)
            if len(kidx):
                np.maximum.at(s0, kidx, -gv0)
            x_init = np.concatenate([x_init, np.maximum(s0, 0.0) + 1e-3])
            lbx_s = np.concatenate([lbx, np.zeros(N)])
            ubx_s = np.concatenate([ubx, np.full(N, np.inf)])
        try:
            sol = slv(x0=x_init, p=P, lbx=lbx_s, ubx=ubx_s, lbg=lbg, ubg=ubg)
        except RuntimeError:
            continue
        u = np.array(sol["x"]).ravel()[:2 * N]
        gv = np.array(gfun(u, P)).ravel()
        viol = max(float(np.max(lbg - gv)), float(np.max(np.where(np.isfinite(ubg), gv - ubg, -np.inf))),
                   float(np.max(np.maximum(lbx - u, u - ubx))))
        if viol <= feas_tol:
            f = float(sol["f"])
            if best is None or f < best[0]:
                best = (f, u, float(np.min(gv[n_rate:])) if len(gv) > n_rate else np.inf)
                solve_step.last_solver = "ipopt" if slv is solver else "sqp"
        elif (SOFT and SOFT_EARLY and slv.stats().get("success", False)
              and float(np.max(np.array(sol["x"]).ravel()[2 * N:])) > SOFT_INFEAS):
            # convergenza con scarto netto: il problema vero e' irrisolvibile (localmente); niente
            # altri avvii, si passa subito al ripiego
            solve_step.soft_infeasible = getattr(solve_step, "soft_infeasible", 0) + 1
            break
    solve_step.last_time = time.perf_counter() - t0
    if best is not None:
        u_full = best[1].reshape(N, 2)
        return u_full[0].copy(), True, u_full, best[2], "optimal"
    return _fallback(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, P, gfun, n_rate, feas_tol, Hb)


solve_step.last_time = 0.0


def _brake_plan(v1, w1, N):
    u = np.zeros((N, 2)); v = v1
    u[0] = (v1, w1)
    for k in range(1, N):
        v = max(0.0, v - A_DEC * DT)
        u[k] = (v, 0.0)
    return u


def _fallback(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, P, gfun, n_rate, tol, Hb=H_MAX):
    """Nessuna soluzione fattibile: frenata massima (rispetta i limiti); se il suo primo
    intervallo viola un vincolo, griglia di primi comandi ammissibili seguiti da frenata,
    controllando il primo intervallo (l'unico nella garanzia per-istante)."""
    J = len(obstacles)

    def first_margin(u):
        gv = np.array(gfun(u.ravel(), P)).ravel()[n_rate:]
        ns = len(SUBS)
        Jb = (len(gv) - Hb * N * ns) // (N * ns)
        gob = gv[:Jb * N * ns].reshape(Jb, N, ns)[:J, 0, :]      # primo intervallo, ostacoli veri
        ghp = gv[Jb * N * ns:].reshape(Hb, N, ns)[:, 0, :]
        return float(min(np.min(gob) if J else np.inf, np.min(ghp) if Hb else np.inf))

    vb = max(0.0, v_mis - A_DEC * DT)
    wb = float(np.clip(0.0, w_mis - ALPHA * DT, w_mis + ALPHA * DT))
    u = _brake_plan(vb, wb, N)
    m = first_margin(u)
    if m >= -tol:
        return u[0].copy(), True, u, m, "fallback_stop"
    best_ok, best_bad = None, None
    for v in np.linspace(max(0.0, v_mis - A_DEC * DT), min(v_max, v_mis + A_ACC * DT), 7):
        for w in np.linspace(max(-omega_max, w_mis - ALPHA * DT), min(omega_max, w_mis + ALPHA * DT), 9):
            u = _brake_plan(v, w, N)
            m = first_margin(u)
            if m >= -tol:
                d = float(np.linalg.norm(rollout(x0, u[:1])[0] - goal))
                if best_ok is None or d < best_ok[0]:
                    best_ok = (d, u, m)
            elif best_bad is None or m > best_bad[0]:
                best_bad = (m, u, m)
    if best_ok is not None:
        return best_ok[1][0].copy(), True, best_ok[1], best_ok[2], "fallback_evasive"
    return best_bad[1][0].copy(), False, best_bad[1], best_bad[2], "infeasible"


# --- scadenza fissa con il piano precedente come riserva (rev. 2026-11-19, nota §3s) -----------

def check_plan(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, u_plan, halfplanes=(), statics=(),
               tol=1e-4):
    """Violazione massima di un piano (N, 2) sui vincoli veri del problema attuale: accelerazione
    dalla velocita' misurata, limiti, arresto terminale, persone e bordi. <= tol: piano valido.
    Restituisce (violazione, P, gfun, n_rate, Hb, ostacoli potati) per l'eventuale ripiego."""
    if PRUNE:
        obstacles = _prune(x0, v_mis, v_max, obstacles, N)
    halfplanes, statics = list(halfplanes)[:H_MAX], list(statics)
    (solver, gfun, lbg, ubg, n_rate), Jb, Hb, Sb = _get(N, len(obstacles), len(halfplanes), len(statics))
    goal = np.asarray(goal, float)
    d = goal - np.asarray(x0[:2], float)
    P = _params(x0, (v_mis, w_mis), goal, obstacles, N, Jb, d / (np.linalg.norm(d) + 1e-9),
                np.asarray(x0[:2], float), halfplanes, statics, Hb, Sb)
    u = np.asarray(u_plan, float).ravel()
    lbx = np.tile([0.0, -omega_max], N).astype(float)
    ubx = np.tile([v_max, omega_max], N).astype(float)
    if TERMINAL_STOP:
        ubx[2 * (N - 1)] = 0.0
    gv = np.array(gfun(u, P)).ravel()
    viol = max(float(np.max(lbg - gv)), float(np.max(np.where(np.isfinite(ubg), gv - ubg, -np.inf))),
               float(np.max(np.maximum(lbx - u, u - ubx))))
    return viol, P, gfun, n_rate, Hb, obstacles


PREV_FALLBACK = os.environ.get("PACER_PREV_FALLBACK", "1") == "1"   # default dal 2026-11-19 (run v30)


class DeadlineSolver:
    """solve_step con scadenza. Il calcolo gira in un thread (CasADi rilascia il GIL); se non
    finisce entro `deadline` secondi si usa il piano accettato al ciclo prima, traslato di un passo
    con uno stop in coda, purche' rispetti i vincoli del problema ATTUALE (nuove predizioni,
    velocita' misurata); altrimenti il ripiego di sempre (frenata o evasione verificate sul primo
    intervallo). Ogni piano finisce da fermo, quindi il piano traslato e' il candidato naturale
    (fattibilita' ricorsiva). Un solo calcolo alla volta: finche' quello in ritardo non termina
    si usa la riserva, e il suo risultato (ormai vecchio) si scarta.
    simulate=True: calcolo sincrono, scadenza applicata a posteriori sul tempo misurato
    (benchmark, dove il mondo non avanza durante il calcolo)."""

    def __init__(self, deadline=0.3, simulate=False):
        self.deadline = float(deadline)
        self.simulate = simulate
        self._thread = None
        self._prev = None            # ultimo piano usato (N, 2)
        self._busy = 0
        self.n_late = 0

    def reset(self):
        self._prev = None

    def __call__(self, x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, **kw):
        if self._thread is not None and self._thread.is_alive():
            return self._backup(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, kw, "busy")
        self._thread = None
        box = {}

        def work():
            box["r"] = solve_step(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, **kw)

        if self.simulate:
            if self._busy > 0:           # il calcolo in ritardo del ciclo prima sta ancora girando
                self._busy -= 1
                return self._backup(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, kw, "busy")
            t_s = time.perf_counter()
            work()
            t_s = time.perf_counter() - t_s
            late = t_s > self.deadline
            if late:                     # cicli successivi in cui quel calcolo occuperebbe ancora il solutore
                self._busy = max(0, int(np.ceil(t_s / DT)) - 1)
        else:
            th = threading.Thread(target=work, daemon=True)
            th.start()
            th.join(self.deadline)
            late = th.is_alive()
            if late:
                self._thread = th
        if late or "r" not in box:
            self.n_late += 1
            return self._backup(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, kw, "late")
        r = box["r"]
        if PREV_FALLBACK and r[4] != "optimal":
            # nessun piano ottimo: prima del ripiego (frenata) il piano precedente traslato, se
            # rispetta i vincoli attuali (rev. 2026-11-19, nota §3s)
            b = self._try_prev(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, kw)
            if b is not None:
                return b[0], b[1], b[2], b[3], "prev"
        self._prev = np.asarray(r[2], float) if r[4] != "infeasible" else None
        return r

    def _try_prev(self, x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, kw):
        if self._prev is None or len(self._prev) != N:
            return None
        hp, stt = kw.get("halfplanes", ()), kw.get("statics", ())
        tol = kw.get("feas_tol", 1e-4)
        u = np.vstack([self._prev[1:], [0.0, 0.0]])
        viol, P, gfun, n_rate, Hb, obs = check_plan(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max,
                                                    u, hp, stt, tol)
        if viol > tol:
            return None
        self._prev = u
        gv = np.array(gfun(u.ravel(), P)).ravel()
        return u[0].copy(), True, u, float(np.min(gv[n_rate:])) if len(gv) > n_rate else np.inf

    def _backup(self, x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, kw, why):
        hp, stt = kw.get("halfplanes", ()), kw.get("statics", ())
        tol = kw.get("feas_tol", 1e-4)
        if self._prev is not None and len(self._prev) == N:
            u = np.vstack([self._prev[1:], [0.0, 0.0]])
            viol, P, gfun, n_rate, Hb, obs = check_plan(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max,
                                                        u, hp, stt, tol)
            if viol <= tol:
                self._prev = u
                gv = np.array(gfun(u.ravel(), P)).ravel()
                return u[0].copy(), True, u, float(np.min(gv[n_rate:])) if len(gv) > n_rate else np.inf, \
                    "deadline_prev"
        else:
            viol, P, gfun, n_rate, Hb, obs = check_plan(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max,
                                                        np.zeros((N, 2)), hp, stt, tol)
        r = _fallback(x0, v_mis, w_mis, goal, obs, N, v_max, omega_max, P, gfun, n_rate, tol, Hb)
        self._prev = np.asarray(r[2], float) if r[4] != "infeasible" else None
        return r[0], r[1], r[2], r[3], "deadline_" + r[4]


if __name__ == "__main__":
    # Precompilazione JIT (una volta, sul robot): python -m pacer_core.unicycle_nmpc_v2 --precompile 6
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompile", type=int, metavar="N", required=True, help="orizzonte (passi)")
    ap.add_argument("--dt", type=float, default=DT)
    a = ap.parse_args()
    set_dt(a.dt)
    JIT = True
    print(f"precompilo N={a.precompile}, dt={a.dt}, taglie {BUCKETS}, cache {JIT_DIR}")
    print(f"fatto in {precompile(a.precompile):.0f} s; errori JIT: {len(_JIT_FAILED)}")
