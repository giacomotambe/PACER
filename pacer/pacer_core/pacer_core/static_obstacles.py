"""Ostacoli STATICI da Nav2 (local costmap) trattati come costo soft
addizionale nell'NMPC — sia nella variante a vincolo rigido (dove il
vincolo rigido resta riservato ai soli pedoni, §Sim2Real "il costmap non e'
un vincolo di disuguaglianza liscio come l'ellisse di Mahalanobis") sia
nella variante a costo soft. Codice NUOVO: nessuna delle due varianti NMPC
originali (simulazione) conosceva ostacoli statici, l'integrazione Nav2 e'
specifica di questo porting ROS2.

Un ostacolo statico e' un punto (cella "lethal"/inflated del costmap
locale, gia' clusterizzata a monte — vedi pacer_nav2_controller, che
interroga direttamente costmap_ros_ in C++, molto piu' efficiente che
serializzare l'intera griglia) con un raggio di sicurezza d_safe (raggio
robot + inflazione residua). Stessa "bolla di comfort" quadratica del
termine di clearance pedonale (mpc/unicycle_nmpc_soft.py::clearance_cost_term),
qui pero' su TUTTO l'orizzonte (0..N-1, non solo k_near): un ostacolo
statico non "sfuma" con l'incertezza al crescere di k come un pedone
predetto, quindi non c'e' motivo di limitarne l'effetto al breve termine.
"""
import numpy as np


def static_clearance_cost_term(pos, static_obstacles):
    """pos: (N,2) posizioni pianificate del robot. static_obstacles: lista
    di dict {"xy": (2,) array-like, "d_safe": float}. Ritorna la somma, su
    TUTTI i passi e TUTTI gli ostacoli, di relu(d_safe - ||pos_k-xy||)^2 —
    zero oltre d_safe, cresce con continuita' avvicinandosi (nessun
    comportamento forzato/discontinuo, stesso principio del termine
    pedonale)."""
    total = 0.0
    for obs in static_obstacles:
        xy = np.asarray(obs["xy"], dtype=float)
        d_safe = float(obs["d_safe"])
        d = np.linalg.norm(pos - xy[None, :], axis=1)
        pen = np.maximum(d_safe - d, 0.0)
        total = total + np.sum(pen ** 2)
    return total
