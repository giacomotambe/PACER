"""D_cal REALE dalle tracce mocap registrate (rev. 2026-11-04).

Input: il CSV scritto da pacer_qualisys_bridge (parametro tracks_log_csv),
colonne tick, stamp, id, x, y, fresh, campionato a dt_out. Il robot (se
presente nel log) viene escluso dai pedoni. Ogni traccia viene spezzata dove
il marker non e' fresco (occlusione): ogni segmento e' un "pedone" separato,
cosi' nessuna finestra attraversa un buco nello storico.

Finestre 8:12 NON sovrapposte (stride = SEQ_LEN, come nella pipeline),
score di Mahalanobis per offset k=1..H con lo stesso predittore usato online,
salvate nel formato di calibration_offline.save_calibration (caricabile dal
parametro calibration_path di pacer_calibration).

Uso (fuori da ROS, sul PC di laboratorio):
    python -m pacer_core.dcal_from_tracks tracks.csv predictor.pt out_cal.npz \
        --robot-id omnicar --n-mpc 6 [--d-near 2.5]

Rev. 2026-11-13 (Mondrian per distanza, nota §3o): con --d-near ogni finestra va
nel gruppo 1 se il pedone primario, all'ultimo istante osservato, e' a meno di
d_near dal robot (posizione del robot dallo stesso CSV, stesso tick), altrimenti
nel gruppo 0; i pool per gruppo sono salvati come pool_g{g}_{k}. Le finestre
senza posa del robot a quel tick restano solo nel pool marginale.
Il D_cal per gruppo va raccolto con il robot IN MOVIMENTO fra le persone
(shadow mode con robot teleoperato o in missione): con il robot fermo il
gruppo vicino non vede la reazione delle persone al robot.
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import torch

from .calibration_offline import quantile_warmstart, save_calibration
from .predictor_utils import compute_cal_pools
from .sfm_generate_dataset import SEQ_LEN, build_windows_from_run
from .social_stgcnn import SocialSTGCNN


def tracks_from_csv(path, robot_id="", dt=0.4, max_speed=3.0):
    """-> {segment_id: [(tick, x, y), ...]} con segmenti di tick consecutivi freschi.
    Un salto piu' veloce di max_speed [m/s] (marker scambiato/riassegnato)
    chiude il segmento: un salto del genere produrrebbe velocita' assurde in
    ingresso al predittore e score enormi o non finiti."""
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["id"] == robot_id:
                continue
            rows[r["id"]].append((int(r["tick"]), float(r["x"]), float(r["y"]), int(r["fresh"])))
    traj = {}
    for pid, seq in rows.items():
        seq.sort()
        seg, n_seg, last = [], 0, None
        for tick, x, y, fresh in seq:
            jump = bool(seg) and np.hypot(x - seg[-1][1], y - seg[-1][2]) > max_speed * dt
            if not fresh or jump or (last is not None and tick != last + 1):
                if seg:
                    traj[f"{pid}#{n_seg}"] = seg
                    n_seg += 1
                seg = []
            if fresh:
                seg.append((tick, x, y))
            last = tick
        if seg:
            traj[f"{pid}#{n_seg}"] = seg
    return traj


def robot_positions_from_csv(path, robot_id):
    """{tick: (x, y)} delle righe fresche del robot."""
    out = {}
    if not robot_id:
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["id"] == robot_id and int(r["fresh"]):
                out[int(r["tick"])] = (float(r["x"]), float(r["y"]))
    return out


def window_groups(windows, robot_xy, d_near, obs_len=8):
    """Gruppo di ogni finestra: 1 = primario a meno di d_near dal robot all'ultimo
    istante osservato, 0 = altrimenti, -1 = posa del robot assente a quel tick."""
    g = []
    for w in windows:
        tick = w["start_frame"] + obs_len - 1
        rb = robot_xy.get(tick)
        if rb is None:
            g.append(-1)
            continue
        p = w["obs_abs"][-1, w["primary_idx"]]
        g.append(int(np.hypot(p[0] - rb[0], p[1] - rb[1]) < d_near))
    return np.array(g, dtype=int)


def _scores_by_window(model, windows, n_mpc):
    """(n_finestre, n_mpc) score di Mahalanobis, nello stesso ordine delle finestre."""
    from .predictor_utils import predict_window
    from .nonconformity import mahalanobis_score
    R = np.full((len(windows), n_mpc), np.nan)
    for i, w in enumerate(windows):
        mu_abs, Sigma = predict_window(model, w)
        truth = w["pred_abs"][:, w["primary_idx"], :]
        for k in range(1, n_mpc + 1):
            R[i, k - 1] = mahalanobis_score(truth[k - 1], mu_abs[k - 1], Sigma[k - 1])
    return R


def build_dcal_grouped(tracks_csv, model_path, n_mpc, robot_id, d_near, alpha=0.1, delta=0.1,
                       hidden_dim=32, pred_len=12):
    """Come build_dcal, piu' i pool per gruppo (Mondrian per distanza dal robot)."""
    if not robot_id:
        raise ValueError("--d-near richiede --robot-id (posizione del robot nel CSV)")
    traj = tracks_from_csv(tracks_csv, robot_id)
    windows = build_windows_from_run(traj, stride=SEQ_LEN, run_tag="mocap")
    if not windows:
        raise RuntimeError(f"nessuna finestra 8:12 completa nel log (servono {SEQ_LEN} campioni)")
    model = SocialSTGCNN(obs_len=8, pred_len=pred_len, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    R = _scores_by_window(model, windows, n_mpc)
    G = window_groups(windows, robot_positions_from_csv(tracks_csv, robot_id), d_near)
    ok = np.all(np.isfinite(R), axis=1)
    if (~ok).any():
        print(f"ATTENZIONE: {int((~ok).sum())} finestre con score non finiti scartate")
    R, G = R[ok], G[ok]
    pools = {k: sorted(R[:, k - 1].tolist()) for k in range(1, n_mpc + 1)}
    group_pools = {(g, k): R[G == g, k - 1] for g in (0, 1) for k in range(1, n_mpc + 1)}
    n0 = len(R)
    q_ws, _ = quantile_warmstart(pools, n0, alpha / n_mpc, delta)
    return n0, pools, q_ws, group_pools, G


def build_dcal(tracks_csv, model_path, n_mpc, robot_id="", alpha=0.1, delta=0.1,
               hidden_dim=32, pred_len=12):
    traj = tracks_from_csv(tracks_csv, robot_id)
    windows = build_windows_from_run(traj, stride=SEQ_LEN, run_tag="mocap")
    if not windows:
        raise RuntimeError("nessuna finestra 8:12 completa nel log: servono tracce di "
                           f"almeno {SEQ_LEN} campioni consecutivi")
    model = SocialSTGCNN(obs_len=8, pred_len=pred_len, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    pools = compute_cal_pools(model, windows, n_mpc)
    n_bad = sum(int(np.sum(~np.isfinite(pools[k]))) for k in pools)
    if n_bad:
        print(f"ATTENZIONE: {n_bad} score non finiti scartati (Sigma degenere?)")
    pools = {k: sorted(float(r) for r in pools[k] if np.isfinite(r)) for k in pools}
    n0 = min(len(v) for v in pools.values())
    pools = {k: v[:n0] if len(v) > n0 else v for k, v in pools.items()}
    q_ws, _ = quantile_warmstart(pools, n0, alpha / n_mpc, delta)
    return n0, pools, q_ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks_csv")
    ap.add_argument("model_path")
    ap.add_argument("out_npz")
    ap.add_argument("--robot-id", default="")
    ap.add_argument("--n-mpc", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--d-near", type=float, default=None,
                    help="Mondrian per distanza: salva anche i pool per gruppo (richiede --robot-id)")
    ap.add_argument("--sigma-pow", type=float, default=None,
                    help="Sigma^g (rev. 2026-11-22): stesso valore del parametro sigma_pow di pacer_predictor")
    ap.add_argument("--sigma-pow-from-k", type=int, default=1,
                    help="primo offset a cui si applica sigma_pow (2 = k=1 invariato)")
    a = ap.parse_args()
    if a.sigma_pow is not None:
        from .predictor_utils import set_sigma_pow
        set_sigma_pow(a.sigma_pow, a.sigma_pow_from_k)
    if a.d_near is not None:
        n0, pools, q_ws, gp, G = build_dcal_grouped(a.tracks_csv, a.model_path, a.n_mpc, a.robot_id,
                                                    a.d_near, a.alpha, a.delta)
        save_calibration(a.out_npz, pools, n0, q_ws, group_pools=gp)
        print(f"D_cal reale per gruppo: {n0} finestre -> {a.out_npz}  (vicine al robot < "
              f"{a.d_near} m: {int((G == 1).sum())}, lontane: {int((G == 0).sum())}, "
              f"senza posa robot: {int((G < 0).sum())})")
        for k in sorted(q_ws):
            a_k = a.alpha / a.n_mpc if k == 1 else 0.05
            qs = [f"{np.quantile(gp[(g, k)], 1 - a_k):.2f}" if len(gp[(g, k)]) else "-" for g in (0, 1)]
            print(f"  k={k}: quantile empirico 1-alpha_k lontani {qs[0]}, vicini {qs[1]}")
        if (G == 1).sum() < 500:
            print("ATTENZIONE: meno di 500 finestre vicine al robot; il warm start del gruppo "
                  "vicino conta poco, BET si adattera' online.")
        return
    n0, pools, q_ws = build_dcal(a.tracks_csv, a.model_path, a.n_mpc, a.robot_id, a.alpha, a.delta)
    save_calibration(a.out_npz, pools, n0, q_ws)
    print(f"D_cal reale: {n0} finestre -> {a.out_npz}")
    for k in sorted(q_ws):
        print(f"  k={k}: q warm-start KL = {q_ws[k]:.3f}  (mediana score {np.median(pools[k]):.3f})")
    if n0 < 2000:
        print("ATTENZIONE: con meno di ~2000 finestre il quantile KL a alpha/n_mpc puo' "
              "restare infinito; raccogliere piu' shadow mode o usare il betting.")


if __name__ == "__main__":
    main()
