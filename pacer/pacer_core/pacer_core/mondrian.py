"""Mondrian conformal per distanza dal robot (rev. 2026-11-13, nota PACER §3m-§3o).

Ogni pedone predetto riceve un gruppo all'istante di emissione t, noto prima di
vedere l'esito (quindi la garanzia per gruppo resta valida):
    g = 1  se ||p_i(t) - robot(t)|| < d_near   (vicino al robot)
    g = 0  altrimenti
Un tracker TUPAC per ogni coppia (gruppo, offset k), confidenza delta/G per
gruppo; copertura >= 1 - alpha - eta_g in ciascun gruppo, e il Teorema 3
diventa J_bar (alpha_1 + max_g eta_{1,g}).

Su SiT (persone reali, robot reale) il gruppo vicino e' quello in cui il
predittore sbaglia di piu': con un solo tracker il miss a k=6 sale a 4.85%
(BET) / 10.35% (static); con i due gruppi 1.40% / 3.62%. In simulazione SFM
(run v12) e' neutro.

Il gruppo e' calcolato in UN solo punto (pacer_predictor) e viaggia nel
messaggio PedestrianPrediction, cosi' calibrazione e MPC usano la stessa
etichetta.
"""
import math

import numpy as np

GROUP_UNKNOWN = 255
MODES = ("none", "dist")


def n_groups(mode):
    if mode not in MODES:
        raise ValueError(f"mondrian_group sconosciuto: '{mode}' ({'|'.join(MODES)})")
    return 1 if mode == "none" else 2


def assign_group(mode, ped_xy, robot_xy, d_near):
    """-> (gruppo, distanza dal robot). robot_xy None = posa ignota."""
    if robot_xy is None:
        d = float("nan")
    else:
        d = float(math.hypot(ped_xy[0] - robot_xy[0], ped_xy[1] - robot_xy[1]))
    if mode == "none":
        return 0, d
    if not math.isfinite(d):
        return GROUP_UNKNOWN, d
    return (1 if d < d_near else 0), d


def unpack_q(ks, q_flat, n_groups_msg, n_mpc):
    """QTupacSnapshot -> array (G, n_mpc); offset mancanti = inf (vincolo
    sempre violato -> fallback, la direzione sicura)."""
    G = max(int(n_groups_msg), 1)
    ks = list(ks)
    q_flat = list(q_flat)
    out = np.full((G, n_mpc), np.inf)
    if len(q_flat) != G * len(ks):
        return out
    for g in range(G):
        for i, k in enumerate(ks):
            if 1 <= k <= n_mpc:
                out[g, k - 1] = q_flat[g * len(ks) + i]
    return out


def q_for_group(q_grp, g):
    """Quantili (n_mpc,) per un pedone del gruppo g. Gruppo ignoto o fuori
    range: massimo sui gruppi (scelta conservativa)."""
    if q_grp.shape[0] == 1:
        return q_grp[0]
    if g == GROUP_UNKNOWN or not (0 <= g < q_grp.shape[0]):
        return q_grp.max(axis=0)
    return q_grp[g]
