"""NMPC uniciclo con costo pedonale SOFT (gaussiane ad area costante, §5
nota "Costo pedonale a gaussiane ad area costante"). Codice NUOVO — NON
modifica mpc/unicycle_nmpc.py: lo importa solo per riusare `rollout`
(identica cinematica uniciclo, nessuna ragione di duplicarla) e la
convenzione DT.

Differenza chiave rispetto alla versione a vincolo rigido: nessun vincolo
di disuguaglianza sui pedoni. Il termine pedonale entra nel costo come
somma (sincronizzata nel tempo: ogni passo k del robot contro il
contributo del pedone ALLO STESSO passo k, §5 nota — mai il campo sommato
su tutti i k, quello serve solo per la visualizzazione). Il problema resta
quindi sempre risolvibile: non esiste la nozione di "infeasible" di
unicycle_nmpc.py (§6 nota) — unico vincolo restano i box su v,omega."""
import numpy as np
from scipy.optimize import minimize

from . import unicycle_nmpc as hard_mpc   # riuso di rollout/DT — NESSUNA modifica al file
from .pedestrian_cost import gaussian_cost_value   # noqa: E402
from .static_obstacles import static_clearance_cost_term   # noqa: E402
# static_clearance_cost_term: ostacoli STATICI (da Nav2 local costmap, §Nav2
# integration nella nota Sim2Real) trattati come costo soft su TUTTO
# l'orizzonte (a differenza di clearance_cost_term/pedoni non si applica
# k_near: un ostacolo statico non "sfuma" con l'incertezza al crescere di k,
# quindi non c'e' motivo di limitarlo al breve termine) — stesso principio
# del termine di clearance pedonale, riusato per coerenza invece di
# reinventare una seconda formula.

DT = hard_mpc.DT


def set_dt(dt):
    global DT
    DT = dt
    hard_mpc.set_dt(dt)


def _unpack(u_flat, N):
    return u_flat.reshape(N, 2)


def pedestrian_cost_term(pos, obstacles_soft):
    """pos: (N,2) posizioni pianificate del robot passo per passo.
    obstacles_soft: lista di dict {mu:(N,2), Sigma_inv:(N,2,2), h:(N,),
    gamma:float}, uno per pedone attivo (mpc/pedestrian_cost.py). Somma,
    per ciascun pedone, SOLO il contributo g_{j,k}(pos_k) allo stesso passo
    k — mai pos_k contro il campo sommato su tutti i k (§5 nota: altrimenti
    si penalizza il robot per posizioni passate/future del pedone rispetto
    al proprio istante)."""
    total = 0.0
    for obs in obstacles_soft:
        total = total + np.sum(gaussian_cost_value(pos, obs["mu"], obs["Sigma_inv"],
                                                      obs["h"], obs["gamma"]))
    return total


def clearance_cost_term(pos, obstacles_soft, d_safe, k_near=2):
    """Termine di personal-space AGGIUNTIVO, in DISTANZA EUCLIDEA reale (non
    di Mahalanobis) — vedi nota "Costo pedonale a gaussiane ad area
    costante" §8.2. Motivazione (trovata analizzando i cicli piu' vicini di
    una missione reale, non per ipotesi): il termine gaussiano g_{j,k}
    e' ad AREA COSTANTE (A=1, come una densita' di probabilita' vera), quindi
    quando la covarianza predetta e' STRETTA (predizione a breve termine,
    molto sicura) il picco e' alto ma STRETTO — a distanza di Mahalanobis
    ~4-5 sigma (che puo' corrispondere a soli 1-1.5 m REALI quando sigma e'
    piccolo) il contributo e' gia' numericamente trascurabile (verificato:
    g~0.02-0.03 anche con h~0.7-0.8), per costruzione matematica di
    qualunque densita' normalizzata. Risultato osservato: un incrocio con
    piu' pedoni a ~1.2-1.6 m REALI di distanza, per ~2 s consecutivi, non
    genera gradiente sufficiente perche' l'ottimizzatore preferisca
    rallentare/fermarsi (verificato: cost_ped cambia in modo trascurabile
    anche aumentando lambda x2 o azzerando w_stage). Il vincolo rigido non
    ha questo problema (soglia dura, non una densita'), il costo soft si'.
    Fix: un termine SEPARATO, calibrato in METRI reali (non in sigma),
    indipendente dalla larghezza della gaussiana calibrata TUPAC — una
    "bolla di comfort" quadratica attiva solo entro d_safe metri dal centro
    predetto, qualunque sia la covarianza:
        sum_{j, k<k_near} relu(d_safe - ||pos_k - mu_{j,k}||)^2
    Zero oltre d_safe, cresce con continuita' (nessun comportamento
    forzato/discontinuo) man mano che ci si avvicina — l'ottimizzatore resta
    libero di scegliere se rallentare, fermarsi o scartare lateralmente,
    a seconda di quale sia piu' economico nel costo TOTALE.

    k_near limita il termine ai primi k_near passi (default 2 -> 0.8s a
    DT=0.4): applicato a TUTTO l'orizzonte causava un effetto collaterale
    (trovato testando la missione completa, non ipotetico): a fine
    missione, con w_goal/w_stage gia' piccoli perche' vicino a B, la
    predizione di un pedone lontano (5-7 m REALI, nessuna minaccia) puo'
    comunque "sconfinare" entro d_safe in qualche passo LONTANO
    dell'orizzonte (k=4,5,6, 1.6-2.4s nel futuro, la parte piu' incerta e
    meno affidabile della predizione — §5/§7 nota: la covarianza cresce
    apposta con k) e il robot restava fermo/oscillava per diversi cicli
    senza motivo reale. Il problema diagnosticato (incrocio REALE con piu'
    pedoni, k=0-1, covarianza STRETTA — vedi sopra) e' specificamente a
    breve termine: limitare il termine a k_near risolve l'effetto
    collaterale mantenendo intatto il beneficio sul caso reale."""
    total = 0.0
    for obs in obstacles_soft:
        d = np.linalg.norm(pos[:k_near] - obs["mu"][:k_near], axis=1)
        pen = np.maximum(d_safe - d, 0.0)
        total = total + np.sum(pen ** 2)
    return total


def cost_fn_soft(u_flat, x0, N, goal, weights, obstacles_soft, lam, d_safe=0.0, w_clear=0.0,
                  static_obstacles=None, w_static=0.0):
    u = _unpack(u_flat, N)
    pos = hard_mpc.rollout(x0, u)
    w_goal, w_stage, w_u, w_du = weights
    J = w_goal * np.sum((pos[-1] - goal) ** 2)
    J += w_stage * np.sum((pos - goal[None, :]) ** 2)
    J += w_u * np.sum(u ** 2)
    du = np.diff(u, axis=0)
    if len(du):
        J += w_du * np.sum(du ** 2)
    J += lam * pedestrian_cost_term(pos, obstacles_soft)
    if w_clear > 0.0:
        J += w_clear * clearance_cost_term(pos, obstacles_soft, d_safe)
    if static_obstacles and w_static > 0.0:
        J += w_static * static_clearance_cost_term(pos, static_obstacles)
    return J


def _total_cost(u_full, x0, N, goal, weights, obstacles_soft, lam, d_safe=0.0, w_clear=0.0,
                 static_obstacles=None, w_static=0.0):
    pos = hard_mpc.rollout(x0, u_full)
    J = cost_fn_soft(u_full.ravel(), x0, N, goal, weights, obstacles_soft, lam, d_safe, w_clear,
                      static_obstacles, w_static)
    cost_ped = float(pedestrian_cost_term(pos, obstacles_soft))
    return float(J), cost_ped


def solve_step_soft(x0, goal, obstacles_soft, N, v_max, omega_max, lam, u_warm=None,
                      weights=(6.0, 0.15, 0.02, 0.05), d_safe=0.0, w_clear=0.0,
                      static_obstacles=None, w_static=0.0):
    """Risolve un ciclo. Nessun vincolo di disuguaglianza (§6 nota): solo i
    box su v,omega -> L-BFGS-B, non SLSQP. Il problema non e' convesso
    (rollout uniciclo non lineare in omega): L-BFGS-B e' un ottimizzatore
    LOCALE, e partendo solo dal warm start (una traiettoria in moto) puo'
    non considerare mai l'opzione di FERMARSI, anche quando fermarsi
    darebbe un costo totale piu' basso (osservazione esplicita dell'utente:
    "considera il fatto che il robot si puo' fermare" — nella versione a
    vincolo rigido questo e' gia' presente come fallback esplicito,
    §3 spec_mpc_unicycle.md; qui mancava). Corretto con un confronto
    esplicito multi-candidato, MAI fidandosi di un solo ottimo locale:
      (1) ottimizzazione dal warm start (comportamento precedente);
      (2) ottimizzazione da init fermo (v=omega=0) — puo' convergere a un
          ottimo locale diverso, es. "fermati, poi riparti";
      (3) arresto ESPLICITO (u=0 per tutto l'orizzonte, nessuna
          ottimizzazione) — sempre disponibile, costo valutato direttamente.
    Si sceglie il candidato con costo TOTALE J piu' basso (non solo il
    termine pedonale): fermarsi che e' peggio in tracking ma molto meglio
    in sicurezza pedonale puo' comunque vincere, esattamente l'effetto
    voluto. Ritorna (u0, u_full, cost_ped, status): status 'optimal' se il
    candidato scelto e' un'ottimizzazione convergente (res.success),
    'suboptimal' se converge male, 'stopped' se vince l'arresto esplicito
    (3) — MAI 'infeasible', qui la nozione non esiste.

    d_safe/w_clear (§8.2 nota, default 0.0 = comportamento invariato):
    attivano il termine addizionale clearance_cost_term, in distanza reale,
    per i casi (osservati in missione, non ipotetici) in cui il termine
    gaussiano calibrato TUPAC da solo non genera abbastanza gradiente per
    preferire il rallentamento quando la covarianza predetta e' stretta —
    vedi il docstring di clearance_cost_term per l'analisi completa."""
    bounds = [(0.0, v_max), (-omega_max, omega_max)] * N
    args = (x0, N, goal, weights, obstacles_soft, lam, d_safe, w_clear, static_obstacles, w_static)

    candidates = []   # (J, u_full, status_candidato)

    if u_warm is not None and len(u_warm) == N:
        x_init_warm = u_warm.copy()
    else:
        x_init_warm = np.tile([0.3 * v_max, 0.0], (N, 1))
    res_warm = minimize(cost_fn_soft, x_init_warm.ravel(), args=args,
                          method="L-BFGS-B", bounds=bounds, options={"maxiter": 100, "ftol": 1e-9})
    u_warm_full = _unpack(res_warm.x, N)
    J_warm, _ = _total_cost(u_warm_full, x0, N, goal, weights, obstacles_soft, lam, d_safe, w_clear,
                             static_obstacles, w_static)
    candidates.append((J_warm, u_warm_full, "optimal" if res_warm.success else "suboptimal"))

    x_init_stop = np.zeros((N, 2))
    res_stop = minimize(cost_fn_soft, x_init_stop.ravel(), args=args,
                          method="L-BFGS-B", bounds=bounds, options={"maxiter": 100, "ftol": 1e-9})
    u_stop_full = _unpack(res_stop.x, N)
    J_stop_opt, _ = _total_cost(u_stop_full, x0, N, goal, weights, obstacles_soft, lam, d_safe, w_clear,
                                 static_obstacles, w_static)
    candidates.append((J_stop_opt, u_stop_full, "optimal" if res_stop.success else "suboptimal"))

    u_zero = np.zeros((N, 2))
    J_zero, _ = _total_cost(u_zero, x0, N, goal, weights, obstacles_soft, lam, d_safe, w_clear,
                             static_obstacles, w_static)
    candidates.append((J_zero, u_zero, "stopped"))

    J_best, u_full, status = min(candidates, key=lambda c: c[0])
    pos = hard_mpc.rollout(x0, u_full)
    cost_ped = float(pedestrian_cost_term(pos, obstacles_soft))
    return u_full[0].copy(), u_full, cost_ped, status
