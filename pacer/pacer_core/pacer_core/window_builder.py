"""Costruzione della finestra 8:12 (obs_abs, primary_idx) per un pedone
PRIMARIO a partire dagli storici GIA' pronti di TUTTI i pedoni tracciati
in un dato istante — stessa logica di mpc/live_pedestrians.py::build_window
(motore SFM) e sfm/generate_dataset.py::build_windows_from_run, qui
generalizzata a una sorgente di tracking qualunque (Qualisys oggi, un
tracker reale in futuro, §Sim2Real): l'unico requisito e' un dict
{id: array (OBS_LEN,2)} di storici gia' completi e allineati nel tempo —
il bufferizzamento/resampling a dt_out e' responsabilita' di chi produce
quel dict (pacer_qualisys_bridge in questo porting)."""
import numpy as np


def build_window(histories, primary_id, obs_len, max_neighbors=12):
    """histories: {id: (obs_len,2) array}, tutte GIA' complete (chi non ha
    ancora obs_len campioni non deve comparire qui — vedi
    pacer_qualisys_bridge, che pubblica solo pedoni "predicibili"). Ritorna
    {"obs_abs": (obs_len,N,2), "primary_idx": 0..N-1} pronto per
    predictor_utils.predict_window, o None se primary_id non e' presente."""
    if primary_id not in histories:
        return None
    px, py = histories[primary_id][-1]
    others = [i for i in histories if i != primary_id]
    others.sort(key=lambda i: (histories[i][-1, 0] - px) ** 2 + (histories[i][-1, 1] - py) ** 2)
    chosen = [primary_id] + others[:max_neighbors]
    N = len(chosen)
    obs_abs = np.zeros((obs_len, N, 2))
    for ni, i in enumerate(chosen):
        obs_abs[:, ni, :] = histories[i]
    return {"obs_abs": obs_abs, "primary_idx": 0}
