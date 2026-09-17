"""Selezione dei K pedoni piu' vicini come ostacoli attivi dell'MPC e
costruzione delle strutture dati consumate dai due solver — stessa logica
(K_NEIGH/CUTOFF_M) di experiment/mpc_ab_mission_threaded.py::build_obstacles
e experiment/mpc_ab_mission_soft.py::build_obstacles_soft nel progetto di
simulazione, qui generalizzata: invece di leggere da un oggetto
`LiveOpenField` (motore SFM) legge da un semplice dict {id: (x,y)} di
posizioni pedone CORRENTI (Qualisys in questo porting, un tracker reale in
futuro — stessa interfaccia, §Sim2Real)."""
import numpy as np

from .pedestrian_cost import build_soft_obstacle


def build_obstacles_hard(preds, positions_now, robot_xy, q2_by_k, k_neigh=6, cutoff_m=7.0,
                           n_mpc=6):
    """preds: {id: (mu_abs (PRED_LEN,2), Sigma (PRED_LEN,2,2))}. positions_now:
    {id: (2,) posizione attuale, stesse id di preds}. Ritorna la lista di
    ostacoli {mu, Sigma, Sigma_inv, q2} consumata da unicycle_nmpc.solve_step."""
    if not preds:
        return []
    ids = list(preds.keys())
    pos_arr = np.array([positions_now[i] for i in ids])
    d = np.linalg.norm(pos_arr - robot_xy[None, :], axis=1)
    order = np.argsort(d)
    chosen = [ids[o] for o in order if d[o] <= cutoff_m][:k_neigh]
    obstacles = []
    for i in chosen:
        mu_abs, Sigma = preds[i]
        mu_k = mu_abs[:n_mpc]
        Sigma_k = Sigma[:n_mpc] + 1e-6 * np.eye(2)[None, :, :]
        Sigma_inv = np.linalg.inv(Sigma_k)
        obstacles.append({"mu": mu_k, "Sigma": Sigma_k, "Sigma_inv": Sigma_inv, "q2": q2_by_k,
                            "id": i})
    return obstacles


def build_obstacles_soft(preds, positions_now, robot_xy, q_by_k, beta=1.0, gamma_exp=1.3,
                           a_gauss=1.0, k_neigh=6, cutoff_m=7.0, n_mpc=6):
    """Come build_obstacles_hard ma costruisce le gaussiane ad area costante
    (mpc/pedestrian_cost.py::build_soft_obstacle) consumate da
    unicycle_nmpc_soft.solve_step_soft."""
    if not preds:
        return []
    ids = list(preds.keys())
    pos_arr = np.array([positions_now[i] for i in ids])
    d = np.linalg.norm(pos_arr - robot_xy[None, :], axis=1)
    order = np.argsort(d)
    chosen = [ids[o] for o in order if d[o] <= cutoff_m][:k_neigh]
    obstacles = []
    for i in chosen:
        mu_abs, Sigma = preds[i]
        mu_k = mu_abs[:n_mpc]
        Sigma_hat_k = Sigma[:n_mpc] + 1e-6 * np.eye(2)[None, :, :]
        obs = build_soft_obstacle(mu_k, Sigma_hat_k, q_by_k, beta=beta, gamma_exp=gamma_exp,
                                    A=a_gauss)
        obs["id"] = i
        obstacles.append(obs)
    return obstacles
