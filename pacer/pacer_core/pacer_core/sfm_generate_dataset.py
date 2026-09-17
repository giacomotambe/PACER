"""
Costruzione di finestre 8:12 (§1 Spec TUPAC-MPC, riusato qui) direttamente
dalle traiettorie simulate SFM (§4 della spec SFM-SFM): niente split di un
pool fisso preesistente, i tre insiemi sono GENERATI da run SFM separate.

Stessa logica di finestre/adiacenza sociale di data/windowing.py del
progetto tupac_mpc (indice frame->pedoni + intersezione, per scalare), qui
riscritta per operare su un singolo run in memoria invece che su un sito
caricato da disco.
"""
import numpy as np
from .sfm_model import simulate_run          # noqa: E402

OBS_LEN = 8
PRED_LEN = 12
SEQ_LEN = OBS_LEN + PRED_LEN


def _ped_frame_dict(peds):
    return {pid: {fi: (x, y) for fi, x, y in seq} for pid, seq in peds.items()}


def build_windows_from_run(traj, stride=SEQ_LEN, max_neighbors=12, run_tag=""):
    """traj: ped_id(locale al run) -> [(frame_idx,x,y), ...]. Ritorna lista
    di finestre (stesso formato di data/windowing.py::build_windows nel
    progetto tupac_mpc), con node_ids resi globalmente unici via run_tag."""
    pfd = _ped_frame_dict(traj)
    frame_to_peds = {}
    for pid, fd in pfd.items():
        for f in fd:
            frame_to_peds.setdefault(f, set()).add(pid)

    windows = []
    for primary_id, seq in traj.items():
        frame_idxs = [fi for fi, x, y in seq]
        fset = set(frame_idxs)
        if not frame_idxs:
            continue
        fmin, fmax = min(frame_idxs), max(frame_idxs)
        for start in range(fmin, fmax - SEQ_LEN + 2, stride):
            needed = list(range(start, start + SEQ_LEN))
            if not all(f in fset for f in needed):
                continue
            obs_frames = needed[:OBS_LEN]
            pred_frames = needed[OBS_LEN:]
            sets = [frame_to_peds.get(f) for f in obs_frames]
            if any(s is None for s in sets):
                continue
            node_id_set = set.intersection(*sets)
            if primary_id not in node_id_set:
                continue
            if len(node_id_set) > max_neighbors + 1:
                px, py = pfd[primary_id][obs_frames[-1]]
                others = [pid for pid in node_id_set if pid != primary_id]
                others.sort(key=lambda pid: (pfd[pid][obs_frames[-1]][0] - px) ** 2
                            + (pfd[pid][obs_frames[-1]][1] - py) ** 2)
                node_ids_local = [primary_id] + others[:max_neighbors]
            else:
                node_ids_local = list(node_id_set)
            N = len(node_ids_local)
            obs_abs = np.zeros((OBS_LEN, N, 2))
            for ti, f in enumerate(obs_frames):
                for ni, pid in enumerate(node_ids_local):
                    obs_abs[ti, ni] = pfd[pid][f]
            pred_mask = np.zeros(N, dtype=bool)
            pred_abs = np.zeros((PRED_LEN, N, 2))
            for ni, pid in enumerate(node_ids_local):
                fd = pfd[pid]
                if all(f in fd for f in pred_frames):
                    pred_mask[ni] = True
                    for ti, f in enumerate(pred_frames):
                        pred_abs[ti, ni] = fd[f]
            if not pred_mask[node_ids_local.index(primary_id)]:
                continue
            windows.append({
                "obs_abs": obs_abs,
                "pred_abs": pred_abs,
                "pred_mask": pred_mask,
                "primary_idx": node_ids_local.index(primary_id),
                "node_ids": [f"{run_tag}_{pid}" for pid in node_ids_local],
                "run_id": run_tag,
                "primary_id_global": f"{run_tag}_{primary_id}",
                "start_frame": start,
            })
    return windows


def social_adjacency(obs_abs, eps=1e-3):
    T, N, _ = obs_abs.shape
    A = np.zeros((T, N, N))
    for t in range(T):
        diff = obs_abs[t, :, None, :] - obs_abs[t, None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1)) + eps
        inv = 1.0 / dist
        np.fill_diagonal(inv, 0.0)
        row_sum = inv.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        A[t] = inv / row_sum
        np.fill_diagonal(A[t], 1.0)
    return A


def generate_pool(sfm_cfg, n_runs, n_peds, seed0, stride=SEQ_LEN, max_neighbors=12,
                   scenario="corridor"):
    """Simula n_runs run SFM indipendenti (seed diversi, nessuno stato
    condiviso, §3 "run indipendenti") e ne estrae le finestre 8:12. Ritorna
    la lista di finestre di TUTTI i run concatenata (ogni run e' comunque
    statisticamente indipendente dagli altri, indipendentemente da quanti
    ne vengono generati qui — la "dipendenza minima" richiesta per
    training/D_cal (§4) viene dal fatto che ogni run e' una simulazione a
    se stante, non da un filtro aggiuntivo su queste finestre)."""
    all_windows = []
    for r in range(n_runs):
        traj = simulate_run(n_peds, sfm_cfg, seed=seed0 + r, scenario=scenario)
        w = build_windows_from_run(traj, stride=stride, max_neighbors=max_neighbors,
                                    run_tag=f"run{seed0 + r}")
        all_windows.extend(w)
    return all_windows
