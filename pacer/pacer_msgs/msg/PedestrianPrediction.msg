# Traiettoria+covarianza predetta (Social-STGCNN, pacer_predictor) per UN
# pedone, passi k=1..n_steps (assoluti, stesso frame del robot/Qualisys).
# rev. 2026-11-11: kind distingue la predizione del modello (calibrata con
# TUPAC) dai pedoni con storico ancora corto (< obs_len), che ricevono un
# vincolo deterministico/empirico e NON entrano nella calibrazione.
uint8 KIND_MODEL=0       # Social-STGCNN, mu + Sigma, vincolo q_k calibrato
uint8 KIND_CONST_VEL=1   # storico >= 2 punti: mu = velocita' costante, disco di raggio radius[k]
uint8 KIND_REACH=2       # storico = 1 punto: mu = posizione attuale, disco solo a k=1
string id
uint8 kind
geometry_msgs/Point current_xy   # ultima posizione OSSERVATA (t=0, non predetta) —
                                    # usata per la selezione dei K vicini/cutoff in
                                    # obstacle_builder, distinta da mu[] (che parte da k=1)
int32 n_steps
geometry_msgs/Point[] mu
# Sigma_k (2x2) per ciascun passo, riga per riga: sigma_flat[4*k : 4*k+4] =
# [Sxx,Sxy,Syx,Syy] del passo k (k=0..n_steps-1).
float64[] sigma_flat
# Solo per kind != KIND_MODEL: raggio [m] del disco in cui puo' trovarsi il
# CENTRO del pedone al passo k (senza il margine L, aggiunto da
# obstacle_builder); 0 = nessun vincolo a quel passo. sigma_flat = identita'.
float64[] radius
# Rev. 2026-11-13 (Mondrian per distanza dal robot, nota PACER §3o): gruppo
# assegnato UNA volta dal predittore all'istante di emissione, cosi'
# calibrazione e MPC usano la stessa etichetta. Con mondrian_group="none"
# vale sempre 0. GROUP_UNKNOWN = posa del robot non disponibile o vecchia:
# la calibrazione non usa lo score, l'MPC usa il quantile piu' grande fra i gruppi.
uint8 GROUP_UNKNOWN=255
uint8 group
float64 d_robot          # distanza da current_xy al robot all'emissione [m]; NaN se ignota
