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
# Liberare la strada (rev. 2026-11-25): costo soft sull'occupazione delle regioni predette delle
# persone, valido ANCHE da fermo (il vincolo rigido e' rilasciato a robot fermo, passive safety).
# Senza questo termine, davanti a un pedone che non devia, fermarsi sulla sua traiettoria e'
# l'ottimo (nessun vincolo violato). Con W_OCC > 0 il robot e' spinto a stare dove la persona non
# sara': fuori dalla regione allargata di OCC_INFL. 0 = spento (standard fino a v52).
W_OCC = float(os.environ.get("PACER_W_OCC", "0.0"))
OCC_INFL = float(os.environ.get("PACER_OCC_INFL", "1.3"))   # fattore sul raggio della regione
# Solo da fermo/lento (rev. 2026-11-25 notte, diagnosi v56): nella folla il costo di occupazione
# applicato anche in moto fa schivare tutti lateralmente (marciapiede: percorso 66 -> 88-93 m). Con
# OCC_V > 0 il termine e' pesato per max(0, 1 - v_k / OCC_V): penalizza il fermarsi (o quasi) nella
# regione predetta, non il passarle accanto in moto (gia' coperto dal vincolo). 0 = sempre (v53).
OCC_V = float(os.environ.get("PACER_OCC_V", "0.0"))
# Solo i primi OCC_KMAX passi (rev. 2026-11-25 notte): con l'arresto terminale ogni piano finisce
# fermo, quindi gli ultimi passi sono sempre "lenti" e il costo spingerebbe il robot a cercarsi un
# punto d'arresto fuori dalle traiettorie altrui (schivate laterali continue). 0 = tutti i passi.
OCC_KMAX = int(os.environ.get("PACER_OCC_KMAX", "0"))
# Solo chi viene incontro (rev. 2026-11-26): il costo di occupazione vale solo per le persone che si
# avvicinano al robot (velocita' verso il robot > OCC_VCLOSE) o quasi ferme (|v| < OCC_VSTILL);
# chi si allontana (stessa direzione davanti, o dall'altra parte) non conta: non verra' addosso a un
# robot fermo, e schivarla allunga il percorso (marciapiede). Un parametro per ostacolo in fondo a P.
OCC_APPROACH = os.environ.get("PACER_OCC_APPROACH", "0") == "1"
OCC_VCLOSE = float(os.environ.get("PACER_OCC_VCLOSE", "0.1"))
OCC_VSTILL = float(os.environ.get("PACER_OCC_VSTILL", "0.3"))
# Vicino al goal (rev. 2026-11-26): il robot DEVE fermarsi sul goal; con persone attorno il costo di
# occupazione lo spingeva via dal punto d'arresto e il robot girava in tondo (marciapiede, fine
# giro: +20-25 m di percorso). Il costo sfuma a zero entro OCC_GOAL_R dal goal (lineare da 2x). 0 = no.
OCC_GOAL_R = float(os.environ.get("PACER_OCC_GOAL_R", "0.0"))
# Passare dietro (rev. 2026-11-25): penalita' per stare DAVANTI alla persona lungo la sua direzione
# di moto predetta e dentro la sua corsia (larghezza ~ raggio della regione allargata). Se la persona
# va in una direzione, il robot pianifica dalla parte opposta (dietro di lei) invece di correrle
# davanti. Sfuma per persone ferme. 0 = spento.
W_AHEAD = float(os.environ.get("PACER_W_AHEAD", "0.0"))
AHEAD_LEN = float(os.environ.get("PACER_AHEAD_LEN", "2.5"))   # [m] quanto avanti conta la corsia
OCC_MS = float(os.environ.get("PACER_OCC_MS", "0.5"))       # costo di occupazione oltre cui provare tutti gli avvii

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
    npar = 3 + 2 + 2 + 4 + J * (2 + N * 2 + N * 4 + N) + 3 * Hn + 3 * Sn + (1 if social else 0) + J
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
    occ_terms = []
    occ_on = P[npar - J:npar] if J else None          # 1 = la persona conta per il costo di occupazione
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
            if W_OCC > 0 and (OCC_KMAX <= 0 or k < OCC_KMAX):
                r_end = pos[k + 1] - mu[:, k]
                m = ca.mtimes([r_end.T, S, r_end]) / (q2[k] * OCC_INFL ** 2 + 1e-9)
                slow = ca.fmax(0, 1 - U[2 * k] / OCC_V) if OCC_V > 0 else 1
                occ_terms.append(occ_on[j] * ca.if_else(q2[k] > 0, 1, 0) * slow * ca.fmax(0, 1 - m) ** 2)
            if W_AHEAD > 0:
                r_end = pos[k + 1] - mu[:, k]
                vk = mu[:, k] - (mu[:, k - 1] if k > 0 else mu0)
                s2 = ca.sumsqr(vk)
                nd = vk / ca.sqrt(s2 + 1e-4)
                fade = s2 / (s2 + (0.1 * DT) ** 2)                   # ~0 per persone ferme
                R2 = q2[k] * OCC_INFL ** 2 / (0.5 * (S[0, 0] + S[1, 1]) + 1e-9)   # raggio^2 medio della regione
                along = nd[0] * r_end[0] + nd[1] * r_end[1]          # > 0: robot davanti alla persona
                lat = nd[0] * r_end[1] - nd[1] * r_end[0]
                occ_terms.append((W_AHEAD / max(W_OCC, 1e-9)) * ca.if_else(q2[k] > 0, 1, 0) * fade
                                 * ca.exp(-lat ** 2 / R2) / (1 + ca.exp(-along / 0.2))
                                 * ca.fmax(0, 1 - along / AHEAD_LEN) ** 2)
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
    if occ_terms:
        J_cost += max(W_OCC, 1e-9) * ca.sum1(ca.vertcat(*occ_terms))
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
    key = (N, Jb, Hb, Sb, SOCIAL, DT, JIT, HYBRID, SOFT, W_SLACK, W_HEAD, W_OCC, OCC_INFL, OCC_V, OCC_KMAX, W_AHEAD, AHEAD_LEN, tuple(sorted(IPOPT_EXTRA.items())))
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
    parts.append(np.array([float(o.get("occ_on", 1.0)) for o in obstacles] + [0.0] * (Jb - len(obstacles))))
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


def _mark_approaching(x0, obstacles):
    """Copia degli ostacoli con occ_on = 1 per chi si avvicina al robot o e' quasi fermo, 0 altrimenti."""
    out = []
    p_r = np.asarray(x0[:2], float)
    for o in obstacles:
        p0 = np.asarray(o["mu0"], float)
        vel = (np.asarray(o["mu"][0], float) - p0) / DT
        d = p_r - p0
        closing = float(vel @ d) / (float(np.linalg.norm(d)) + 1e-9)
        on = 1.0 if (float(np.linalg.norm(vel)) < OCC_VSTILL or closing > OCC_VCLOSE) else 0.0
        o2 = dict(o); o2["occ_on"] = on
        out.append(o2)
    return out


def _occ(x0, u, obstacles, N):
    """Occupazione del piano u (N, 2): somma di relu(1 - r'Sr / (q2 * OCC_INFL^2))^2 sugli
    ostacoli e sugli offset, come il termine W_OCC del costo."""
    pos = rollout(x0, u)
    c = 0.0
    for o in obstacles:
        mu = np.asarray(o["mu"][:N], float); Si = np.asarray(o["Sigma_inv"][:N], float)
        q2 = np.asarray(o["q2"][:N], float)
        prev = np.asarray(o["mu0"], float)
        for k in range(N):
            if q2[k] > 0 and np.isfinite(q2[k]):
                r = pos[k] - mu[k]
                if W_OCC > 0 and (OCC_KMAX <= 0 or k < OCC_KMAX):
                    slow = max(0.0, 1.0 - float(u[k, 0]) / OCC_V) if OCC_V > 0 else 1.0
                    c += float(o.get("occ_on", 1.0)) * slow * max(0.0, 1.0 - float(r @ Si[k] @ r) / (q2[k] * OCC_INFL ** 2)) ** 2
                if W_AHEAD > 0:
                    vk = mu[k] - prev; s2 = float(vk @ vk)
                    nd = vk / np.sqrt(s2 + 1e-4); fade = s2 / (s2 + (0.1 * DT) ** 2)
                    R2 = q2[k] * OCC_INFL ** 2 / (0.5 * (Si[k][0, 0] + Si[k][1, 1]) + 1e-9)
                    al = float(nd @ r); lat = float(nd[0] * r[1] - nd[1] * r[0])
                    c += (W_AHEAD / max(W_OCC, 1e-9)) * fade * np.exp(-lat ** 2 / R2) / (1 + np.exp(-al / 0.2))                         * max(0.0, 1 - al / AHEAD_LEN) ** 2
            prev = mu[k]
    return c


# --- scelta della strategia: aspettare o aggirare (rev. 2026-11-25) ----------------------------
# L'orizzonte dell'MPC (N*DT = 2.4 s) e' troppo corto per capire se conviene rallentare e lasciar
# passare una persona o aggirarla: ogni ciclo si simulano su STRAT_T secondi alcune strategie
# semplici (via laterale a sinistra/destra di diverse ampiezze x attesa/rallentamento iniziale),
# con le persone che proseguono la traiettoria predetta, e si sceglie quella che arriva prima
# senza entrare nelle regioni predette (e senza fermarsi dentro). Il goal dell'MPC diventa la
# posizione della strategia scelta dopo STRAT_TSUB secondi; vincoli e garanzie non cambiano.
STRATEGY = os.environ.get("PACER_STRATEGY", "0") == "1"
STRAT_T = float(os.environ.get("PACER_STRAT_T", "8.0"))       # [s] orizzonte della simulazione
STRAT_DT = 0.2
STRAT_TSUB = float(os.environ.get("PACER_STRAT_TSUB", "3.2"))  # [s] sotto-goal per l'MPC
STRAT_LAT = (0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4)            # [m] scarto laterale della via
STRAT_HOLD = ((0.0, 1.0), (1.2, 0.4), (2.4, 0.4), (1.2, 0.0), (2.4, 0.0))   # (durata [s], frazione di v_max)
STRAT_W_IN = float(os.environ.get("PACER_STRAT_W_IN", "30.0"))     # [s per s] in moto dentro una regione
STRAT_W_STILL = float(os.environ.get("PACER_STRAT_W_STILL", "3.0"))  # [s per s] fermo dentro una regione
STRAT_W_STOP = float(os.environ.get("PACER_STRAT_W_STOP", "0.3"))    # [s per s] fermo (in generale)
STRAT_HYST = float(os.environ.get("PACER_STRAT_HYST", "0.4"))        # [s] bonus alla strategia precedente
STRAT_VIA_D = 3.0                    # [m] distanza della via lungo la direzione del goal
_STRAT_PREV = {"c": (0.0, 0)}


def _ped_futures(obstacles, N, T, dt):
    """Posizioni (J, M, 2) e raggi (J, M) delle persone a t = dt, 2dt, ..., T: interpolazione della
    predizione (passo DT) e prosecuzione a velocita' costante oltre l'orizzonte."""
    M = int(round(T / dt))
    ts = dt * np.arange(1, M + 1)
    P_, R_ = [], []
    for o in obstacles:
        mu = np.vstack([np.asarray(o["mu0"], float)[None], np.asarray(o["mu"][:N], float)])
        q2 = np.asarray(o["q2"][:N], float)
        Si = np.asarray(o["Sigma_inv"][:N], float)
        rad = np.sqrt(np.maximum(np.where(np.isfinite(q2), q2, 0.0), 0.0)
                      / (0.5 * (Si[:, 0, 0] + Si[:, 1, 1]) + 1e-9)) * OCC_INFL
        if not np.any(rad > 0):
            continue
        last = rad[rad > 0][-1]
        rad = np.where(rad > 0, rad, last)
        tk = DT * np.arange(0, N + 1)
        vel = (mu[-1] - mu[-2]) / DT
        px = np.interp(ts, tk, mu[:, 0]) + np.maximum(ts - tk[-1], 0) * vel[0]
        py = np.interp(ts, tk, mu[:, 1]) + np.maximum(ts - tk[-1], 0) * vel[1]
        rr = np.interp(ts, tk[1:], rad)
        P_.append(np.stack([px, py], 1)); R_.append(rr)
    if not P_:
        return np.zeros((0, M, 2)), np.zeros((0, M))
    return np.array(P_), np.array(R_)


def select_strategy(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, halfplanes=()):
    """Ritorna (sotto-goal (2,), (lat, hold), costo, controlli (N, 2) della strategia a passo DT)."""
    x0 = np.asarray(x0, float); goal = np.asarray(goal, float)
    PP, RR = _ped_futures(obstacles, N, STRAT_T, STRAT_DT)
    cands = [(l, h) for l in STRAT_LAT for h in range(len(STRAT_HOLD))]
    C = len(cands); M = int(round(STRAT_T / STRAT_DT))
    dg = goal - x0[:2]; dist0 = float(np.linalg.norm(dg))
    e = dg / (dist0 + 1e-9); nrm = np.array([-e[1], e[0]])
    via = np.array([x0[:2] + e * min(STRAT_VIA_D, 0.6 * dist0) + nrm * l for l, _ in cands])
    hold_t = np.array([STRAT_HOLD[h][0] for _, h in cands]); hold_v = np.array([STRAT_HOLD[h][1] for _, h in cands])
    px = np.full(C, x0[0]); py = np.full(C, x0[1]); th = np.full(C, x0[2])
    v = np.full(C, max(v_mis, 0.0)); w = np.full(C, float(w_mis))
    passed = np.array([l == 0.0 for l, _ in cands])
    cost = np.zeros(C); t_reach = np.full(C, np.nan)
    traj = np.zeros((C, M, 2)); ctrl = np.zeros((C, M, 2))
    for m in range(M):
        t = (m + 1) * STRAT_DT
        tgt = np.where(passed[:, None], goal[None], via)
        dx = tgt[:, 0] - px; dy = tgt[:, 1] - py
        err = (np.arctan2(dy, dx) - th + np.pi) % (2 * np.pi) - np.pi
        w_c = np.clip(2.5 * err, -omega_max, omega_max)
        w = np.clip(w_c, w - ALPHA * STRAT_DT, w + ALPHA * STRAT_DT)
        v_c = np.where(t <= hold_t, hold_v * v_max, v_max) * np.clip(np.cos(err), 0.0, 1.0)
        dgo = np.hypot(goal[0] - px, goal[1] - py)
        v_c = np.minimum(v_c, np.sqrt(2 * A_DEC * np.maximum(dgo - 0.1, 0.0)))
        v = np.where(v_c >= v, np.minimum(v_c, v + A_ACC * STRAT_DT), np.maximum(v_c, v - A_DEC * STRAT_DT))
        px = px + v * np.cos(th) * STRAT_DT; py = py + v * np.sin(th) * STRAT_DT; th = th + w * STRAT_DT
        traj[:, m] = np.stack([px, py], 1); ctrl[:, m] = np.stack([v, w], 1)
        # via superata quando la proiezione lungo e arriva a 0.3 m da quella della via
        passed |= ((np.stack([px, py], 1) - via) @ e) > -0.3
        live = np.isnan(t_reach)
        dgo = np.hypot(goal[0] - px, goal[1] - py)
        if len(PP):
            d = np.hypot(px[:, None] - PP[None, :, m, 0], py[:, None] - PP[None, :, m, 1])   # (C, J)
            inside = np.maximum(0.0, 1.0 - d / RR[None, :, m]).max(1)
            moving = v > EPS_V
            cost += live * STRAT_DT * inside * np.where(moving, STRAT_W_IN, STRAT_W_STILL)
        for nx_, ny_, dd in halfplanes:
            cost += live * STRAT_DT * STRAT_W_IN * np.maximum(0.0, dd - (nx_ * px + ny_ * py))
        cost += live * STRAT_DT * STRAT_W_STOP * (v < EPS_V) * (dgo > 0.5)
        t_reach = np.where(live & (dgo < 0.3), t, t_reach)
    dgo = np.hypot(goal[0] - px, goal[1] - py)
    t_arr = np.where(np.isnan(t_reach), STRAT_T + dgo / max(v_max, 1e-3), t_reach)
    total = cost + t_arr
    prev = _STRAT_PREV["c"]
    if prev in cands:
        total[cands.index(prev)] -= STRAT_HYST
    i = int(np.argmin(total))
    _STRAT_PREV["c"] = cands[i]
    ms = int(round(STRAT_TSUB / STRAT_DT)) - 1
    sub = goal if (cands[i] == (0.0, 0) or dist0 < STRAT_VIA_D) else traj[i, ms]
    step = int(round(DT / STRAT_DT))
    u = ctrl[i, step - 1::step][:N].copy()
    if len(u) < N:
        u = np.vstack([u, np.zeros((N - len(u), 2))])
    select_strategy.last = dict(choice=cands[i], cost=float(total[i]), straight=float(total[0]), sub=sub)
    return sub, cands[i], float(total[i]), u


select_strategy.last = None


# --- selettore veloce (rev. 2026-11-25 sera) ---------------------------------------------------
# Stessa idea di select_strategy, con quattro risparmi:
# 1. solo le persone che possono interagire col robot entro STRAT_T (distanza dalla loro traiettoria
#    predetta meno il raggio della regione < raggiungibile dal robot + 0.5 m);
# 2. nessuna persona rilevante (o il rettilineo senza attesa non ha penalita'): si sceglie il
#    rettilineo senza simulare le altre strategie (caso piu' frequente);
# 3. 28 strategie invece di 35 (tolto il rallentamento lungo, (2.4 s, 0.4));
# 4. passo di simulazione STRATF_DT = 0.4 s con il controllo di distanza anche a meta' passo
#    (stessa risoluzione spaziale del passo 0.2 s per il contatto).
STRAT_FAST = os.environ.get("PACER_STRAT_FAST", "0") == "1"
STRAT_STATIC_PAD = float(os.environ.get("PACER_STRAT_STATIC_PAD", "0.3"))   # [m] margine attorno agli statici
STRATF_DT = 0.4
STRATF_HOLD = ((0.0, 1.0), (1.2, 0.4), (1.2, 0.0), (2.4, 0.0))
STRATF_MARGIN = 0.5
# Taratura per la folla (rev. 2026-11-25 sera, dopo v54): oltre l'orizzonte del predittore (N*DT)
# la persona e' solo prolungata in linea retta, quindi (a) i conflitti pesano meno col tempo,
# fattore exp(-(t - N*DT) / STRAT_TAU), (b) la regione non cresce oltre STRAT_R_FAR, e (c) il
# selettore cambia strategia solo se il rettilineo ha una penalita' oltre STRAT_ACT.
STRAT_TAU = float(os.environ.get("PACER_STRAT_TAU", "1e9"))      # [s] 1e9 = nessuno sconto (v54)
STRAT_R_FAR = float(os.environ.get("PACER_STRAT_R_FAR", "1e9"))  # [m] 1e9 = come la regione a N (v54)
STRAT_ACT = float(os.environ.get("PACER_STRAT_ACT", "1e-9"))     # [s] soglia di attivazione (v54: ~0)
# Corsie fisse (rev. 2026-11-25 notte, diagnosi v56): le vie laterali erano relative alla posizione
# attuale del robot, quindi "+0.8 m" scelto due volte di fila diventava +1.6 m (deriva) e nella
# folla il selettore cambiava strategia ogni 2-3 cicli (42-161 cambi per episodio, zig-zag). Con
# STRAT_LANES le vie sono corsie fisse rispetto alla retta del percorso (path_org, path_dir), piu'
# la corsia attuale del robot; cambiare corsia costa STRAT_SWITCH secondi per metro, cambiare tipo
# di attesa STRAT_SWITCH_H secondi.
STRAT_LANES = os.environ.get("PACER_STRAT_LANES", "0") == "1"
STRAT_SWITCH = float(os.environ.get("PACER_STRAT_SWITCH", "0.8"))
STRAT_SWITCH_H = float(os.environ.get("PACER_STRAT_SWITCH_H", "0.5"))
# Chi sorpassa da dietro (rev. 2026-11-25 notte): una persona dietro al robot che va nella stessa
# direzione piu' veloce lo raggiunge comunque; spostarsi per lei rallenta il robot (marciapiede).
# Con STRAT_IGNORE_BEHIND il selettore non la considera (resta il vincolo dell'MPC e il filtro).
STRAT_IGNORE_BEHIND = os.environ.get("PACER_STRAT_IGNORE_BEHIND", "0") == "1"


def _strat_sim(x0, v_mis, w_mis, goal, cands, holds, PP, RR, halfplanes, v_max, omega_max, dt, M):
    """Simula le strategie cands = [(lat, i_hold)] per M passi di dt. PP (J, 2M, 2), RR (J, 2M):
    persone a meta' passo e a fine passo. Ritorna (costo totale, traiettorie (C, M, 2), controlli)."""
    goal = np.asarray(goal, float)
    C = len(cands)
    dg = goal - x0[:2]; dist0 = float(np.linalg.norm(dg))
    e = dg / (dist0 + 1e-9); nrm = np.array([-e[1], e[0]])
    lat = np.array([c[0] for c in cands])
    via = x0[:2][None] + e[None] * min(STRAT_VIA_D, 0.6 * dist0) + nrm[None] * lat[:, None]
    hold_t = np.array([holds[c[1]][0] for c in cands]); hold_v = np.array([holds[c[1]][1] for c in cands])
    px = np.full(C, x0[0]); py = np.full(C, x0[1]); th = np.full(C, x0[2])
    v = np.full(C, max(v_mis, 0.0)); w = np.full(C, float(w_mis))
    passed = lat == 0.0
    cost = np.zeros(C); t_reach = np.full(C, np.nan)
    traj = np.zeros((C, M, 2)); ctrl = np.zeros((C, M, 2))
    J = len(PP)
    for m in range(M):
        t = (m + 1) * dt
        tx = np.where(passed, goal[0], via[:, 0]); ty = np.where(passed, goal[1], via[:, 1])
        err = (np.arctan2(ty - py, tx - px) - th + np.pi) % (2 * np.pi) - np.pi
        w = np.clip(np.clip(2.5 * err, -omega_max, omega_max), w - ALPHA * dt, w + ALPHA * dt)
        dgo = np.hypot(goal[0] - px, goal[1] - py)
        v_c = np.minimum(np.where(t <= hold_t, hold_v * v_max, v_max) * np.clip(np.cos(err), 0.0, 1.0),
                         np.sqrt(2 * A_DEC * np.maximum(dgo - 0.1, 0.0)))
        v = np.where(v_c >= v, np.minimum(v_c, v + A_ACC * dt), np.maximum(v_c, v - A_DEC * dt))
        hx = px + 0.5 * v * np.cos(th) * dt; hy = py + 0.5 * v * np.sin(th) * dt
        px = px + v * np.cos(th) * dt; py = py + v * np.sin(th) * dt; th = th + w * dt
        traj[:, m, 0] = px; traj[:, m, 1] = py; ctrl[:, m, 0] = v; ctrl[:, m, 1] = w
        passed |= ((px - via[:, 0]) * e[0] + (py - via[:, 1]) * e[1]) > -0.3
        live = np.isnan(t_reach)
        dgo = np.hypot(goal[0] - px, goal[1] - py)
        if J:
            ins = 0.0
            for (qx, qy), col in (((hx, hy), 2 * m), ((px, py), 2 * m + 1)):
                d = np.hypot(qx[:, None] - PP[None, :, col, 0], qy[:, None] - PP[None, :, col, 1])
                ins = np.maximum(ins, np.maximum(0.0, 1.0 - d / RR[None, :, col]).max(1))
            cost += live * dt * ins * np.where(v > EPS_V, STRAT_W_IN, STRAT_W_STILL)
        for nx_, ny_, dd in halfplanes:
            cost += live * dt * STRAT_W_IN * np.maximum(0.0, dd - (nx_ * px + ny_ * py))
        cost += live * dt * STRAT_W_STOP * (v < EPS_V) * (dgo > 0.5)
        t_reach = np.where(live & (dgo < 0.3), t, t_reach)
    dgo = np.hypot(goal[0] - px, goal[1] - py)
    t_arr = np.where(np.isnan(t_reach), M * dt + dgo / max(v_max, 1e-3), t_reach)
    return cost, cost + t_arr, traj, ctrl


def _strat_sim_vec(x0, v_mis, goal, cands, holds, PP, RR, halfplanes, v_max, omega_max, dt, M, wt=None,
                   via=None):
    """Versione senza ciclo sui passi: ogni strategia e' il percorso a due tratti x0 -> via -> goal
    percorso con il suo profilo di velocita' (attesa/rallentamento, poi v_max, limiti di
    accelerazione); la rotazione iniziale verso il primo tratto costa |angolo| / omega_max secondi
    di ritardo. Controllo delle persone a passo dt/2. Stesse uscite di _strat_sim."""
    goal = np.asarray(goal, float)
    C = len(cands)
    dg = goal - x0[:2]; dist0 = float(np.linalg.norm(dg))
    e = dg / (dist0 + 1e-9); nrm = np.array([-e[1], e[0]])
    lat = np.array([c[0] for c in cands], float)
    if via is None:
        via = np.where((lat == 0.0)[:, None], goal[None],
                       x0[:2][None] + e[None] * min(STRAT_VIA_D, 0.6 * dist0) + nrm[None] * lat[:, None])
    seg1 = via - x0[:2][None]; L1 = np.linalg.norm(seg1, axis=1)
    seg2 = goal[None] - via; L2 = np.linalg.norm(seg2, axis=1)
    d1 = seg1 / (L1[:, None] + 1e-9); d2 = seg2 / (L2[:, None] + 1e-9)
    ang = np.abs((np.arctan2(d1[:, 1], d1[:, 0]) - x0[2] + np.pi) % (2 * np.pi) - np.pi)
    t_turn = ang / max(omega_max, 1e-3)
    # profilo di velocita' per ogni attesa (H righe), passo dt/2
    h2 = 0.5 * dt; K = 2 * M
    ts = h2 * np.arange(1, K + 1)
    key = (round(max(v_mis, 0.0), 2), float(v_max), h2, K, tuple(holds), A_ACC, A_DEC)
    vh = _VPROF.get(key)
    if vh is None:                         # profili per ogni attesa, in cache per velocita' iniziale
        H = len(holds)
        vh = np.zeros((H, K)); v = np.full(H, key[0])
        ht = np.array([h[0] for h in holds]); hv = np.array([h[1] for h in holds])
        for k in range(K):
            vc = np.where(ts[k] <= ht, hv * v_max, v_max)
            v = np.where(vc >= v, np.minimum(vc, v + A_ACC * h2), np.maximum(vc, v - A_DEC * h2))
            vh[:, k] = v
        if len(_VPROF) > 4096:
            _VPROF.clear()
        _VPROF[key] = vh
    hidx = np.array([c[1] for c in cands])
    vel = vh[hidx]                                                    # (C, K)
    # ritardo di rotazione: velocita' nulla per i primi t_turn secondi (approssimazione)
    vel = np.where(ts[None] < t_turn[:, None], 0.0, vel)
    s = np.cumsum(vel, 1) * h2                                       # arco percorso
    Ltot = L1 + L2
    s = np.minimum(s, Ltot[:, None])
    on1 = s <= L1[:, None]
    pos = np.where(on1[..., None], x0[:2][None, None] + d1[:, None] * s[..., None],
                   via[:, None] + d2[:, None] * (s - L1[:, None])[..., None])     # (C, K, 2)
    reached = s >= Ltot[:, None] - 0.3
    t_reach = np.where(reached.any(1), ts[np.argmax(reached, 1)], np.nan)
    live = ~np.concatenate([np.zeros((C, 1), bool), reached[:, :-1]], 1)    # passi prima dell'arrivo
    still = vel < EPS_V
    cost = np.zeros(C)
    if len(PP):
        d = np.hypot(pos[:, None, :, 0] - PP[None, :, :, 0], pos[:, None, :, 1] - PP[None, :, :, 1])  # (C, J, K)
        ins = np.maximum(0.0, 1.0 - d / RR[None]).max(1)                                               # (C, K)
        if wt is not None:
            ins = ins * wt[None]
        cost += (h2 * ins * np.where(still, STRAT_W_STILL, STRAT_W_IN) * live).sum(1)
    for nx_, ny_, dd in halfplanes:
        cost += (h2 * STRAT_W_IN * np.maximum(0.0, dd - (nx_ * pos[..., 0] + ny_ * pos[..., 1])) * live).sum(1)
    cost += (h2 * STRAT_W_STOP * still * ~reached * live).sum(1)
    dgo = Ltot - s[:, -1]
    t_arr = np.where(np.isnan(t_reach), K * h2 + dgo / max(v_max, 1e-3), t_reach)
    # controlli a passo dt (velocita' lungo il percorso; omega verso la direzione del tratto)
    traj = pos[:, 1::2]
    vv = vel[:, 1::2]
    hd = np.where(on1[:, 1::2], np.arctan2(d1[:, 1], d1[:, 0])[:, None], np.arctan2(d2[:, 1], d2[:, 0])[:, None])
    prev_h = np.concatenate([np.full((C, 1), x0[2]), hd[:, :-1]], 1)
    ww = np.clip(((hd - prev_h + np.pi) % (2 * np.pi) - np.pi) / dt, -omega_max, omega_max)
    ctrl = np.stack([vv, ww], 2)
    return cost, cost + t_arr, traj, ctrl


STRAT_VEC = os.environ.get("PACER_STRAT_VEC", "1") == "1"
_VPROF = {}   # selettore veloce: simulazione senza ciclo


def select_strategy_fast(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max, halfplanes=(),
                         path_dir=None, path_org=None, statics=()):
    """Come select_strategy (stessa interfaccia), piu' veloce: vedi il commento sopra."""
    x0 = np.asarray(x0, float); goal = np.asarray(goal, float)
    dt = STRATF_DT; M = int(round(STRAT_T / dt))
    if STRAT_IGNORE_BEHIND and obstacles:
        hd = np.array([np.cos(x0[2]), np.sin(x0[2])])
        keep_o = []
        for o in obstacles:
            p0 = np.asarray(o["mu0"], float); vel = (np.asarray(o["mu"][0], float) - p0) / DT
            behind = float((p0 - x0[:2]) @ hd) < -0.3
            same = float(vel @ hd) > 0.3
            if not (behind and same):
                keep_o.append(o)
        obstacles = keep_o
    PP, RR = _ped_futures(obstacles, N, STRAT_T, dt / 2)           # (J, 2M, ...)
    tsh = 0.5 * dt * np.arange(1, 2 * M + 1)
    if statics:
        # ostacoli statici (costmap) come persone ferme di raggio d_safe + STRAT_STATIC_PAD (rev. 2026-11-26,
        # riproduzione NUC: senza, il selettore mandava il robot a incastrarsi dietro un pilastro)
        x0a = np.asarray(x0[:2], float)
        st = np.array([(sx, sy, ds) for sx, sy, ds in statics if ds > 0 and np.hypot(sx - x0a[0], sy - x0a[1]) < 8.0])
        if len(st):
            Ps = np.repeat(st[:, None, :2], len(tsh), 1)
            Rs = np.repeat((st[:, 2:3] + STRAT_STATIC_PAD), len(tsh), 1)
            PP = np.concatenate([PP, Ps], 0) if len(PP) else Ps
            RR = np.concatenate([RR, Rs], 0) if len(RR) else Rs
    far = tsh > N * DT
    if len(RR) and STRAT_R_FAR < 1e8:
        RR = np.where(far[None], np.minimum(RR, STRAT_R_FAR), RR)
    wt = np.where(far, np.exp(-(tsh - N * DT) / STRAT_TAU), 1.0) if STRAT_TAU < 1e8 else None
    if len(PP):
        ts = 0.5 * dt * np.arange(1, 2 * M + 1)
        reach = v_max * ts + STRATF_MARGIN                          # robot: al massimo v_max * t
        dmin = np.hypot(PP[:, :, 0] - x0[0], PP[:, :, 1] - x0[1]) - RR
        keep = (dmin < reach[None]).any(1)
        PP, RR = PP[keep], RR[keep]
    dist0 = float(np.linalg.norm(goal - x0[:2]))
    straight = (0.0, 0)
    stats = dict(n_peds=int(len(PP)), n_cands=1)

    def done(ch, sub, u, tot):
        _STRAT_PREV["c"] = ch
        select_strategy.last = dict(choice=ch, cost=tot, sub=sub, **stats)
        return sub, ch, tot, u

    def plan_of(ctrl_i):
        u = ctrl_i[:N].copy() if abs(DT - dt) < 1e-9 else ctrl_i[int(round(DT / dt)) - 1::int(round(DT / dt))][:N].copy()
        return np.vstack([u, np.zeros((N - len(u), 2))]) if len(u) < N else u

    if not len(PP) and not halfplanes:
        return done(straight, goal, None, 0.0)
    if STRAT_VEC:
        def sim(x0_, vm, wm, g, c, h, P_, R_, hp, vmax, om, dt_, M_, via=None):
            return _strat_sim_vec(x0_, vm, g, c, h, P_, R_, hp, vmax, om, dt_, M_, wt, via)
    else:
        sim = _strat_sim
    pen, tot, traj, ctrl = sim(x0, v_mis, w_mis, goal, [straight], STRATF_HOLD, PP, RR, halfplanes,
                                      v_max, omega_max, dt, M)
    if pen[0] <= STRAT_ACT:
        return done(straight, goal, plan_of(ctrl[0]), float(tot[0]))
    if STRAT_LANES and STRAT_VEC:
        pdir = np.asarray(path_dir if path_dir is not None else (goal - x0[:2]) / (dist0 + 1e-9), float)
        porg = np.asarray(path_org if path_org is not None else x0[:2], float)
        if float((goal - x0[:2]) @ pdir) < 0:                # giro di ritorno: retta percorsa al contrario
            pdir = -pdir
        nrm = np.array([-pdir[1], pdir[0]])
        s0 = float((x0[:2] - porg) @ pdir); c0 = float((x0[:2] - porg) @ nrm)
        s_goal = float((goal - porg) @ pdir)
        lanes = sorted(set([round(l, 2) for l in STRAT_LAT] + [round(c0, 2)]))
        cands = [(l, h) for l in lanes for h in range(len(STRATF_HOLD))]
        s_via = min(s0 + STRAT_VIA_D, s_goal)
        via = np.array([porg + pdir * s_via + nrm * l for l, _ in cands])
        stats["n_cands"] = len(cands)
        pen, tot, traj, ctrl = sim(x0, v_mis, w_mis, goal, cands, STRATF_HOLD, PP, RR, halfplanes,
                                   v_max, omega_max, dt, M, via=via)
        prev = _STRAT_PREV["c"]
        prev_lane = prev[0] if prev != straight else c0     # dal rettilineo: si parte dalla corsia attuale
        tot += STRAT_SWITCH * np.abs(np.array([l for l, _ in cands]) - prev_lane)
        tot += STRAT_SWITCH_H * (np.array([h for _, h in cands]) != prev[1])
    else:
        cands = [(l, h) for l in STRAT_LAT for h in range(len(STRATF_HOLD))]
        stats["n_cands"] = len(cands)
        pen, tot, traj, ctrl = sim(x0, v_mis, w_mis, goal, cands, STRATF_HOLD, PP, RR, halfplanes,
                                   v_max, omega_max, dt, M)
        prev = _STRAT_PREV["c"]
        if prev in cands:
            tot[cands.index(prev)] -= STRAT_HYST
    i = int(np.argmin(tot))
    ms = max(int(round(STRAT_TSUB / dt)) - 1, 0)
    sub = goal if (cands[i] == straight or dist0 < STRAT_VIA_D) else traj[i, ms]
    return done(cands[i], sub, plan_of(ctrl[i]), float(tot[i]))


# Retta delle corsie senza percorso globale (rev. 2026-11-26, ROS): il nodo ROS non passa la retta
# del percorso e il goal e' un punto di lookahead che si sposta col robot. Le corsie del selettore
# devono restare fisse, quindi si tiene un'ancora (origine, direzione) e la si rifa' solo quando la
# direzione verso il goal ruota di piu' di LANE_REANCHOR_DEG o il goal esce di piu' di
# LANE_REANCHOR_LAT metri dalla retta (curva del percorso).
LANE_REANCHOR_DEG = float(os.environ.get("PACER_LANE_REANCHOR_DEG", "25"))
LANE_REANCHOR_LAT = float(os.environ.get("PACER_LANE_REANCHOR_LAT", "1.0"))
_LANE_ANCHOR = {}


def _lane_anchor(x0, goal):
    p = np.asarray(x0[:2], float); g = np.asarray(goal, float)
    d = g - p; nd = float(np.linalg.norm(d))
    if nd < 1e-6:
        return np.array([1.0, 0.0]), p
    d = d / nd
    a = _LANE_ANCHOR
    if a:
        cosang = float(d @ a["dir"])
        rg = g - a["org"]
        lat = abs(float(a["dir"][0] * rg[1] - a["dir"][1] * rg[0]))     # prodotto vettoriale 2D
        if cosang > np.cos(np.radians(LANE_REANCHOR_DEG)) and lat < LANE_REANCHOR_LAT:
            return a["dir"], a["org"]
    a["dir"], a["org"] = d, p
    return d, p


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
    goal_true = goal.copy()
    if path_dir is None and STRATEGY and STRAT_LANES:
        path_dir, path_org = _lane_anchor(x0, goal)
    if path_dir is None:
        d = goal - np.asarray(x0[:2], float)
        path_dir = d / (np.linalg.norm(d) + 1e-9)
    if path_org is None:
        path_org = np.asarray(x0[:2], float)
    if STRATEGY:
        prev_c = _STRAT_PREV["c"]
        if STRAT_FAST:
            goal, choice, _, u_strat = select_strategy_fast(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max,
                                                            list(halfplanes)[:H_MAX], path_dir, path_org, list(statics))
        else:
            goal, choice, _, u_strat = select_strategy(x0, v_mis, w_mis, goal, obstacles, N, v_max, omega_max,
                                                       list(halfplanes)[:H_MAX])
        if u_strat is not None and (choice != prev_c or u_warm is None):
            u_warm = u_strat     # cambio di strategia: si parte dal suo piano
    if W_OCC > 0 and (OCC_APPROACH or OCC_GOAL_R > 0):
        obstacles = _mark_approaching(x0, obstacles) if OCC_APPROACH else [dict(o) for o in obstacles]
        if OCC_GOAL_R > 0:
            dgoal = float(np.linalg.norm(np.asarray(goal_true, float) - np.asarray(x0[:2], float)))
            fade = float(np.clip((dgoal - OCC_GOAL_R) / OCC_GOAL_R, 0.0, 1.0))
            for o in obstacles:
                o["occ_on"] = float(o.get("occ_on", 1.0)) * fade
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
            # liberare la strada: se il piano trovato resta nelle regioni predette, si provano anche
            # le curve a sinistra e a destra (rompe la simmetria del frontale)
            if not ((W_OCC > 0 or W_AHEAD > 0) and MULTISTART == "lazy" and _occ(x0, best[1].reshape(N, 2), obstacles, N) > OCC_MS):
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
