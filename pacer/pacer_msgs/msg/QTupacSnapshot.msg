# Ultimo snapshot di q_tupac(k) disponibile dal worker di calibrazione
# online (pacer_calibration) — MAI un'attesa sul ricalcolo, stessa
# semantica di mpc/calibration_worker.py::snapshot_q nella simulazione.
std_msgs/Header header
int32[] k
# Rev. 2026-11-13 (Mondrian): con n_groups = G > 1, q ha G*len(k) elementi in
# ordine gruppo-maggiore, q[g*len(k) + i] = quantile del gruppo g all'offset k[i].
# n_groups = 0 o 1: un solo gruppo, comportamento precedente.
float64[] q
int32 n_groups
float64 d_near           # soglia [m] del gruppo "vicino al robot" (informativa)
int32 n_processed
int32 max_backlog
float64 staleness_mean_ms
