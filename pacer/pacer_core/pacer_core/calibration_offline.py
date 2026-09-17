"""Costruzione di D_cal e del warm-start di q_tupac(k) — richiamabile su
richiesta (servizio ROS2 RecalibrateTupac in pacer_calibration), MAI
implicita nel nodo online (§Sim2Real "i codici di training e calibrazione
devono restare disponibili e richiamabili una o piu' volte su richiesta
dell'utente, non nascosti nel nodo online").

Oggi (§Sim2Real "aggiornamento — ground truth pedoni") D_cal e' ancora
generato da simulazione SFM (generate_pool, stesso codice della pipeline di
sviluppo) — nessun dato reale raccolto col Qualisys e' stato ancora usato
per ricalibrare. build_dcal_from_qualisys_log() sotto e' il punto di
estensione gia' predisposto per quando servira' (§Sim2Real, criticita' sul
domain shift simulato/reale): stesso formato di finestre in output
(build_windows_from_run), quindi tutto il resto della pipeline (compute_cal_pools,
quantile warm-start) resta identico qualunque sia la sorgente."""
import numpy as np

from .sfm_generate_dataset import generate_pool, OBS_LEN, PRED_LEN   # noqa: E402
from .predictor_utils import compute_cal_pools                         # noqa: E402
from .quantile_tupac import HFunction, quantile_from_sorted_pool         # noqa: E402


def build_dcal_from_sfm(model, sfm_cfg, n_runs_cal, n_peds_scene, seed0, H,
                          scenario="open_field"):
    """Come mpc_batch_compare.py::main (sezione D_cal): n_runs_cal run SFM
    indipendenti, generati UNA volta. Ritorna (n0, pools) — pools[k] gia'
    ordinato, pronto per il warm-start del quantile."""
    cal_windows = generate_pool(sfm_cfg, n_runs_cal, n_peds_scene, seed0=seed0,
                                  scenario=scenario)
    pools = compute_cal_pools(model, cal_windows, H)
    n0 = len(cal_windows)
    return n0, pools


def build_dcal_from_qualisys_log(model, log_windows, H):
    """Punto di estensione (non ancora usato in produzione, §Sim2Real):
    log_windows nello STESSO formato di sfm_generate_dataset.build_windows_from_run
    (obs_abs/pred_abs/pred_mask/primary_idx), ma costruito da una
    registrazione Qualisys reale invece che da simulazione SFM — richiede
    un tool offline separato (fuori scope qui) che tagli lo stream Qualisys
    registrato in finestre 8:12 con verita' futura nota (gia' osservata nel
    log). Stessa compute_cal_pools() di build_dcal_from_sfm, quindi stesso
    formato di pools in output."""
    pools = compute_cal_pools(model, log_windows, H)
    n0 = len(log_windows)
    return n0, pools


def quantile_warmstart(pools, n0, alpha_win, delta, h_kind="heavy_tail"):
    h = HFunction(h_kind)
    N_MPC = max(pools.keys())
    q_warmstart = {k: quantile_from_sorted_pool(np.array(pools[k]), n0, alpha_win, delta, h,
                                                  t0=0, bound="kl") for k in range(1, N_MPC + 1)}
    return q_warmstart, h


def save_calibration(path, pools, n0, q_warmstart):
    np.savez(path, n0=n0, q_warmstart_k=np.array(sorted(q_warmstart)),
              q_warmstart_v=np.array([q_warmstart[k] for k in sorted(q_warmstart)]),
              **{f"pool_{k}": np.array(v) for k, v in pools.items()})


def load_calibration(path):
    d = np.load(path)
    n0 = int(d["n0"])
    ks = d["q_warmstart_k"].tolist()
    vs = d["q_warmstart_v"].tolist()
    q_warmstart = {int(k): float(v) for k, v in zip(ks, vs)}
    pools = {int(k): d[f"pool_{k}"].tolist() for k in ks}
    return n0, pools, q_warmstart
