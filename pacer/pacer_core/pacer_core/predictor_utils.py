"""Inferenza del predittore (predict_window) e costruzione del pool di
calibrazione (compute_cal_pools) — stesso codice, stessa semantica di
calibration/online_loop.py nel progetto di simulazione, qui con import
relativi (package pacer_core) invece degli hack su sys.path usati negli
script offline. Riusato identico dal nodo pacer_predictor (inferenza online)
e dal nodo pacer_calibration (costruzione di D_cal, §Sim2Real "codici di
training/calibrazione sempre disponibili su richiesta")."""
import numpy as np
import torch

from .sfm_generate_dataset import social_adjacency   # noqa: E402
from .train_predictor import to_velocity               # noqa: E402
from .social_stgcnn import SocialSTGCNN                  # noqa: E402
from .nonconformity import mahalanobis_score               # noqa: E402


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
