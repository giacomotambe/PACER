"""Training di Social-STGCNN su finestre generate da run SFM indipendenti
(§1, §4 spec SFM-SFM). Stessa logica di training del progetto tupac_mpc
(forward+backward per finestra, batch accumulato), qui su un dataset
interamente in memoria (nessun caricamento da disco)."""
import os
import time
import json
import numpy as np
import torch

from .sfm_generate_dataset import social_adjacency, OBS_LEN, PRED_LEN   # noqa: E402
from .social_stgcnn import SocialSTGCNN, gaussian_nll                     # noqa: E402

# ROOT ora e' passato esplicitamente da chi chiama train() (nodo ROS2
# pacer_calibration, servizio TrainPredictor) — niente piu' assunzione di
# una struttura di cartelle "../results" fissa come nello script offline
# originale.


def to_velocity(obs_abs):
    v = np.zeros_like(obs_abs)
    v[1:] = obs_abs[1:] - obs_abs[:-1]
    v[0] = v[1] if len(v) > 1 else 0
    return v


def train(train_windows, cfg, out_path, log_path, seed=None):
    """out_path/log_path ora OBBLIGATORI (passati dal chiamante, tipicamente
    il nodo pacer_calibration in risposta al servizio TrainPredictor) —
    nello script offline originale avevano un default relativo a una
    struttura di progetto fissa che qui non esiste piu'."""
    seed = seed if seed is not None else cfg.get("seed", 42)

    torch.manual_seed(seed)
    model = SocialSTGCNN(obs_len=OBS_LEN, pred_len=PRED_LEN,
                          hidden_dim=cfg["predictor"]["hidden_dim"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["predictor"]["lr"])

    epochs = cfg["predictor"]["epochs"]
    batch_size = 8
    rng = np.random.default_rng(seed)
    log = []
    t0 = time.time()
    for ep in range(epochs):
        order = rng.permutation(len(train_windows))
        epoch_loss = 0.0
        n_batches = 0
        opt.zero_grad()
        acc = 0
        for idx in order:
            w = train_windows[idx]
            obs_abs = w["obs_abs"]
            last_pos = obs_abs[-1]
            X = torch.tensor(to_velocity(obs_abs), dtype=torch.float32)
            A = torch.tensor(social_adjacency(obs_abs), dtype=torch.float32)
            target = torch.tensor(w["pred_abs"] - last_pos[None, :, :], dtype=torch.float32)
            mask = torch.tensor(w["pred_mask"])
            if mask.sum() == 0:
                continue
            raw = model(X, A)
            mu, sx, sy, rho = SocialSTGCNN.raw_to_gaussian(raw)
            loss = gaussian_nll(mu, sx, sy, rho, target, mask=mask)
            (loss / batch_size).backward()
            epoch_loss += loss.item()
            n_batches += 1
            acc += 1
            if acc == batch_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
                acc = 0
        if acc > 0:
            opt.step()
            opt.zero_grad()
        avg = epoch_loss / max(n_batches, 1)
        log.append({"epoch": ep, "nll": avg, "t": time.time() - t0})
        print(f"epoch {ep:3d}  NLL medio train = {avg:.4f}  ({time.time()-t0:.1f}s)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"modello salvato in {out_path}")
    return model
