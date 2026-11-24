"""Inferenza del predittore (predict_window) e costruzione del pool di
calibrazione (compute_cal_pools) — stesso codice, stessa semantica di
calibration/online_loop.py nel progetto di simulazione, qui con import
relativi (package pacer_core) invece degli hack su sys.path usati negli
script offline. Riusato identico dal nodo pacer_predictor (inferenza online)
e dal nodo pacer_calibration (costruzione di D_cal, §Sim2Real "codici di
training/calibrazione sempre disponibili su richiesta")."""
import os

import numpy as np
import torch

from .sfm_generate_dataset import social_adjacency   # noqa: E402
from .train_predictor import to_velocity               # noqa: E402
from .social_stgcnn import SocialSTGCNN                  # noqa: E402
from .nonconformity import mahalanobis_score               # noqa: E402


# rev. 2026-11-22 (nota PACER §3s, prova B / run v44): Sigma_k -> Sigma_k^g (autovalori alla g).
# Le ellissi del predittore sono troppo eccentriche: con g = 0.5 l'area a pari copertura cala del
# 20-40% e il tempo di missione del 4-14%. Si applica qui, in UN solo punto, cosi' predittore online,
# D_cal (compute_cal_pools, dcal_from_tracks) e MPC usano la stessa forma. Il D_cal DEVE essere
# costruito con lo stesso g usato online. Default 1.0 = comportamento precedente.
SIGMA_POW = float(os.environ.get("PACER_SIGMA_POW", "1.0"))
# rev. 2026-11-22 (run v45, stress): g applicato solo dagli offset k >= SIGMA_POW_FROM_K (1-based).
# Con 2, k=1 (l'offset della garanzia, Teorema 3) usa la Sigma del predittore invariata.
SIGMA_POW_FROM_K = int(os.environ.get("PACER_SIGMA_POW_FROM_K", "1"))


def set_sigma_pow(g, from_k=None):
    global SIGMA_POW, SIGMA_POW_FROM_K
    SIGMA_POW = float(g)
    if from_k is not None:
        SIGMA_POW_FROM_K = int(from_k)


def sigma_pow(S, g):
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-9) ** g
    return V @ (w[..., None] * np.swapaxes(V, -1, -2))


def predict_window(model, window):
    """window: {"obs_abs": (OBS_LEN,N,2), "primary_idx": int}. Ritorna
    mu_abs (PRED_LEN,2) e Sigma (PRED_LEN,2,2) — posizione assoluta e
    covarianza predette per il pedone primario, un solo forward pass."""
    obs_abs = window["obs_abs"]
    last_pos = obs_abs[-1]
    X = torch.tensor(to_velocity(obs_abs), dtype=torch.float32)
    A = torch.tensor(social_adjacency(obs_abs), dtype=torch.float32)
    with torch.no_grad():
        raw = model(X, A)
        mu, sx, sy, rho = SocialSTGCNN.raw_to_gaussian(raw)
        Sigma = SocialSTGCNN.sigma_matrix(sx, sy, rho)
    p = window["primary_idx"]
    mu_abs = mu[:, p, :].numpy() + last_pos[p][None, :]
    Sigma_np = Sigma[:, p, :, :].numpy()
    if SIGMA_POW != 1.0:
        k0 = max(SIGMA_POW_FROM_K, 1) - 1
        Sigma_np = Sigma_np.copy()
        Sigma_np[k0:] = sigma_pow(Sigma_np[k0:], SIGMA_POW)
    return mu_abs, Sigma_np


def compute_cal_pools(model, cal_windows, H):
    """Score di Mahalanobis realizzati (predizione vs verita' nota) su un
    set di finestre — usato SOLO offline per costruire D_cal (la verita'
    futura deve essere gia' nota, quindi non e' applicabile online su
    Qualisys se non a posteriori su un log registrato)."""
    pools = {k: [] for k in range(1, H + 1)}
    for w in cal_windows:
        mu_abs, Sigma = predict_window(model, w)
        truth = w["pred_abs"][:, w["primary_idx"], :]
        for k in range(1, H + 1):
            r = mahalanobis_score(truth[k - 1], mu_abs[k - 1], Sigma[k - 1])
            pools[k].append(r)
    for k in pools:
        pools[k] = sorted(pools[k])
    return pools
