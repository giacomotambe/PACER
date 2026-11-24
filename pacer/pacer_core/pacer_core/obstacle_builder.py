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


def build_obstacles_hard(preds, positions_now, robot_xy, q_by_k, k_neigh=6, cutoff_m=7.0,
                           n_mpc=6, clearance_l=0.6, q_by_id=None):
    """preds: {id: (mu_abs (PRED_LEN,2), Sigma (PRED_LEN,2,2))}. positions_now:
    {id: (2,) posizione attuale, stesse id di preds}. Ritorna la lista di
    ostacoli {mu, Sigma, Sigma_inv, q2} consumata da unicycle_nmpc.solve_step.

    Rev. 2026-11-04: q2 = c^2 con c_jk = q_k + L/sqrt(lambda_min(Sigma_jk))
    (raggio del robot + corpo del pedone in unita' di Mahalanobis, Lemma
    geometrico del paper). Prima q2 = q_k^2, cioe' robot puntiforme. Un q_k
    non finito (calibrazione assente) rende il vincolo sempre violato ->
    fallback: e' la direzione sicura.

    Rev. 2026-11-13 (Mondrian): q_by_id = {id: (n_mpc,)} quantili del gruppo del
    pedone; se assente o senza quell'id si usa q_by_k (uguale per tutti)."""
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
        lam_min = np.linalg.eigvalsh(Sigma_k)[:, 0]
        q_i = q_by_id.get(i, q_by_k) if q_by_id else q_by_k
        c = np.asarray(q_i, dtype=float)[:n_mpc] + clearance_l / np.sqrt(lam_min)
        obstacles.append({"mu": mu_k, "Sigma": Sigma_k, "Sigma_inv": Sigma_inv, "q2": c ** 2,
                            "id": i, "mu0": np.asarray(positions_now[i], dtype=float)})
    return obstacles


def build_obstacles_soft(preds, positions_now, robot_xy, q_by_k, beta=1.0, gamma_exp=1.3,
                           a_gauss=1.0, k_neigh=6, cutoff_m=7.0, n_mpc=6, q_by_id=None):
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
        q_i = q_by_id.get(i, q_by_k) if q_by_id else q_by_k
        obs = build_soft_obstacle(mu_k, Sigma_hat_k, np.asarray(q_i, dtype=float)[:n_mpc],
                                    beta=beta, gamma_exp=gamma_exp, A=a_gauss)
        obs["id"] = i
        obstacles.append(obs)
    return obstacles


def build_unpredicted_hard(unpred, robot_xy, cutoff_m=7.0, n_mpc=6, clearance_l=0.6, current=None):
    """Rev. 2026-11-11. unpred: {id: (mu (n,2), radius (n,))} per i pedoni con
    storico corto (pacer_core/unpredicted.py). Vincolo a disco sul centro del
    robot, raggio L + radius_k (0 = nessun vincolo a quel passo). Tutti quelli
    entro cutoff_m, NON limitati ai k_neigh vicini (come nel benchmark)."""
    obstacles = []
    for i, (mu, radius) in unpred.items():
        mu = np.asarray(mu, dtype=float)
        if np.linalg.norm(mu[0] - robot_xy) > cutoff_m + float(radius[0]):
            continue
        r = np.asarray(radius[:n_mpc], dtype=float)
        q2 = np.where(r > 0, (clearance_l + r) ** 2, 0.0)
        eye = np.repeat(np.eye(2)[None], n_mpc, 0)
        mu0 = np.asarray(current[i], dtype=float) if current and i in current else mu[0]
        obstacles.append({"mu": mu[:n_mpc], "Sigma": eye, "Sigma_inv": eye,
                          "q2": q2, "id": i, "unpredicted": True, "mu0": mu0})
    return obstacles


def unpredicted_as_static(unpred, robot_xy, cutoff_m=7.0, clearance_l=0.6):
    """Modalita' soft: il costo soft accetta solo gaussiane calibrate, quindi i
    pedoni con storico corto entrano come ostacoli statici (costo di
    clearance) alla posizione prevista a k=1, d_safe = L + radius_1."""
    out = []
    for i, (mu, radius) in unpred.items():
        mu = np.asarray(mu, dtype=float)
        if np.linalg.norm(mu[0] - robot_xy) <= cutoff_m + float(radius[0]):
            out.append({"xy": (float(mu[0][0]), float(mu[0][1])), "d_safe": clearance_l + float(radius[0])})
    return out
