"""NMPC uniciclo con vincolo rigido di evitamento (§1/§3 spec_mpc_unicycle.md).

Single-shooting: le variabili di decisione sono i comandi u=(v_i,omega_i)_{i=0..N-1},
lo stato e' ottenuto per rollout esplicito della cinematica. Risolto con SLSQP
(scipy) — piccolo problema (2N variabili, <=K*N vincoli), jacobiane per
differenze finite (di default in SLSQP), warm start dal ciclo precedente.
"""
import numpy as np
from scipy.optimize import minimize


def rollout(x0, u):
    """x0=(px,py,theta). u: (N,2) = (v,omega). Ritorna positions (N,2) (p_1..p_N,
    NON include x0) e thetas (N,)."""
    N = u.shape[0]
    px, py, th = x0
    pos = np.empty((N, 2))
    for i in range(N):
        v, om = u[i]
        px = px + v * np.cos(th) * DT
        py = py + v * np.sin(th) * DT
        th = th + om * DT
        pos[i] = (px, py)
    return pos


DT = 0.4  # sovrascritto da set_dt() se serve altro passo


def set_dt(dt):
    global DT
    DT = dt


def _unpack(u_flat, N):
    return u_flat.reshape(N, 2)


def cost_fn(u_flat, x0, N, goal, weights, static_obstacles=None, w_static=0.0):
    u = _unpack(u_flat, N)
    pos = rollout(x0, u)
    w_goal, w_stage, w_u, w_du = weights
    J = w_goal * np.sum((pos[-1] - goal) ** 2)
    J += w_stage * np.sum((pos - goal[None, :]) ** 2)
    J += w_u * np.sum(u ** 2)
    du = np.diff(u, axis=0)
    if len(du):
        J += w_du * np.sum(du ** 2)
    if static_obstacles and w_static > 0.0:
        from .static_obstacles import static_clearance_cost_term
        J += w_static * static_clearance_cost_term(pos, static_obstacles)
    return J


def _ellipse_constraint_values(u_flat, x0, N, obstacles):
    """obstacles: lista di dict {mu: (N,2), Sigma_inv: (N,2,2), q2: (N,)} —
    uno per pedone, gia' allineati passo-per-passo k=1..N. Ritorna vettore
    (n_obstacles*N,) di g_jk = (p_k-mu_jk)^T Sinv_jk (p_k-mu_jk) - q2_jk >= 0."""
    u = _unpack(u_flat, N)
    pos = rollout(x0, u)  # (N,2)
    vals = []
    for obs in obstacles:
        diff = pos - obs["mu"]                      # (N,2)
        # (N,1,2) @ (N,2,2) @ (N,2,1) -> (N,)
        tmp = np.einsum('ni,nij->nj', diff, obs["Sigma_inv"])
        maha2 = np.einsum('nj,nj->n', tmp, diff)
        vals.append(maha2 - obs["q2"])
    return np.concatenate(vals) if vals else np.array([1.0])  # nessun ostacolo -> sempre feasible


def _feasible(u_full, x0, N, obstacles, tol):
    """Verifica ESPLICITA del vincolo rigido sulla soluzione candidata — non ci
    si fida del solo flag res.success di SLSQP (che riflette convergenza KKT,
    non garantisce soddisfacimento del vincolo entro una tolleranza stretta,
    specie da un punto iniziale molto infeasible): senza questo controllo il
    "vincolo rigido" non sarebbe davvero rigido (bug, non solo un limite)."""
    if not obstacles:
        return True, float("inf")
    margins = _ellipse_constraint_values(u_full.ravel(), x0, N, obstacles)
    m = float(np.min(margins))
    return m >= -tol, m


def solve_step(x0, goal, obstacles, N, v_max, omega_max, u_warm=None,
                weights=(6.0, 0.15, 0.02, 0.05), feas_tol=1e-4,
                static_obstacles=None, w_static=0.0):
    """Risolve un ciclo di NMPC. Ritorna (u0, ok, u_full, margin, status):
    u0=(v,omega) del primo passo da applicare, u_full=(N,2) la sequenza
    pianificata (warm start del ciclo successivo), margin=min margine del
    vincolo rigido sulla soluzione EFFETTIVAMENTE applicata, status in
    {"optimal","fallback_stop","infeasible"}. Se SLSQP converge ma la
    soluzione viola il vincolo rigido oltre feas_tol, NON viene accettata:
    si prova l'arresto in posizione (u=0); se anche questo viola il vincolo
    (nessuna azione del robot puo' soddisfarlo in questo ciclo — orizzonte/
    dinamica insufficienti), lo si applica comunque come scelta piu' sicura
    disponibile ma si segnala "infeasible" (mai un successo silenzioso, §3
    spec_mpc_unicycle.md)."""
    if u_warm is not None and len(u_warm) == N:
        x_init = u_warm.copy()
    else:
        x_init = np.tile([0.3 * v_max, 0.0], (N, 1))
    x_init_flat = x_init.ravel()

    bounds = [(0.0, v_max), (-omega_max, omega_max)] * N

    constraints = []
    if obstacles:
        constraints.append({
            "type": "ineq",
            "fun": lambda uf: _ellipse_constraint_values(uf, x0, N, obstacles),
        })

    res = minimize(cost_fn, x_init_flat, args=(x0, N, goal, weights, static_obstacles, w_static),
                    method="SLSQP", bounds=bounds, constraints=constraints,
                    options={"maxiter": 60, "ftol": 1e-4})

    if res.success:
        u_full_opt = _unpack(res.x, N)
        ok, margin = _feasible(u_full_opt, x0, N, obstacles, feas_tol)
        if ok:
            return u_full_opt[0].copy(), True, u_full_opt, margin, "optimal"

    u_zero = np.zeros((N, 2))
    ok_zero, margin_zero = _feasible(u_zero, x0, N, obstacles, feas_tol)
    status = "fallback_stop" if ok_zero else "infeasible"
    return np.array([0.0, 0.0]), ok_zero, u_zero, margin_zero, status
