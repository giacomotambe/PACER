"""Pedoni con storico ancora corto (< obs_len): vincolo senza predittore
(rev. 2026-11-11, stessa logica di 13_closed_loop_benchmark/run_benchmark.py
con PACER_UNPRED=cv, il default del benchmark).

- storico >= 2 punti -> velocita' costante: mu_k = p_t + k (p_t - p_{t-1}),
  disco di raggio CV_Q[k-1] attorno a mu_k su tutti gli offset k = 1..n;
- storico = 1 punto  -> raggiungibilita': disco di raggio v_cap * dt attorno
  alla posizione attuale, solo a k = 1 (Assunzione 3 del paper).

I raggi riguardano il CENTRO del pedone; il margine L (raggio robot + corpo
pedone) lo aggiunge obstacle_builder. CV_Q e' il quantile 0.999 dell'errore
della velocita' costante a k passi misurato su 120 run SFM open-field: e' un
vincolo EMPIRICO, non coperto dal Teorema 3. Sull'hardware va rimisurato su
tracce reali (log CSV del bridge) prima di fidarsene.
"""
import numpy as np

KIND_MODEL, KIND_CONST_VEL, KIND_REACH = 0, 1, 2
CV_Q_SFM = (0.47, 1.00, 1.52, 2.08, 2.72, 3.38)


def unpredicted_constraint(history, n_steps, mode="cv", cv_q=CV_Q_SFM, v_cap=2.5, dt=0.4):
    """history: (m,2) posizioni a cadenza dt, m >= 1. Ritorna (kind, mu (n,2),
    radius (n,)). mode: 'cv' (velocita' costante su tutti gli offset), 'cv1'
    (solo k=1) o 'reach' (sempre il disco di raggiungibilita' a k=1)."""
    h = np.asarray(history, dtype=float).reshape(-1, 2)
    radius = np.zeros(n_steps)
    if mode in ("cv", "cv1") and len(h) >= 2:
        if len(cv_q) < n_steps:
            raise ValueError(f"cv_q ha {len(cv_q)} valori, servono almeno n_steps={n_steps}")
        v = h[-1] - h[-2]
        mu = h[-1][None, :] + np.arange(1, n_steps + 1)[:, None] * v[None, :]
        nk = n_steps if mode == "cv" else 1
        radius[:nk] = np.asarray(cv_q[:nk], dtype=float)
        return KIND_CONST_VEL, mu, radius
    mu = np.repeat(h[-1][None, :], n_steps, axis=0)
    radius[0] = v_cap * dt
    return KIND_REACH, mu, radius
