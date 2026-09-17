# Ultimo snapshot di q_tupac(k) disponibile dal worker di calibrazione
# online (pacer_calibration) — MAI un'attesa sul ricalcolo, stessa
# semantica di mpc/calibration_worker.py::snapshot_q nella simulazione.
std_msgs/Header header
int32[] k
float64[] q
int32 n_processed
int32 max_backlog
float64 staleness_mean_ms
