# Traiettoria+covarianza predetta (Social-STGCNN, pacer_predictor) per UN
# pedone, passi k=1..n_steps (assoluti, stesso frame del robot/Qualisys).
string id
geometry_msgs/Point current_xy   # ultima posizione OSSERVATA (t=0, non predetta) —
                                    # usata per la selezione dei K vicini/cutoff in
                                    # obstacle_builder, distinta da mu[] (che parte da k=1)
int32 n_steps
geometry_msgs/Point[] mu
# Sigma_k (2x2) per ciascun passo, riga per riga: sigma_flat[4*k : 4*k+4] =
# [Sxx,Sxy,Syx,Syy] del passo k (k=0..n_steps-1).
float64[] sigma_flat
