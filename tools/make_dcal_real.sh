#!/bin/bash
# D_cal REALE dalle tracce Qualisys di una prova (rev. 2026-11-24). Da lanciare DENTRO il container.
#
# Uso:  tools/make_dcal_real.sh <prova> <robot_body> [uscita.npz] [predittore.pt]
#   <prova>       cartella in /home/ros/bags/runs (o percorso di un tracks.csv), es. run_20261124_101500
#   <robot_body>  nome del robot in QTM (lo stesso di robot_body:= al launch)
#   uscita        default /models/cal_real.npz
#   predittore    default /models/predictor_sim.pt (deve essere lo stesso usato online)
#
# La prova va registrata col robot IN MOVIMENTO fra le persone (teleguidato o in missione) e con
#   ros2 launch pacer_bringup pacer_bringup.launch.py robot_body:=<nome> run_name:=auto
# Poi si usa il D_cal con:  ros2 launch pacer_bringup pacer_bringup.launch.py ... calibration:=/models/cal_real.npz
set -e
if [ $# -lt 2 ]; then sed -n '2,13p' "$0"; exit 1; fi
RUN="$1"; ROBOT="$2"; OUT="${3:-/models/cal_real.npz}"; MODEL="${4:-/models/predictor_sim.pt}"
if [ -d "$RUN" ]; then CSV="$RUN/tracks.csv"
elif [ -f "$RUN" ]; then CSV="$RUN"
else CSV="/home/ros/bags/runs/$RUN/tracks.csv"; fi
[ -f "$CSV" ] || { echo "tracce non trovate: $CSV"; exit 1; }
[ -f "$MODEL" ] || { echo "predittore non trovato: $MODEL"; exit 1; }
source /opt/ros/${ROS_DISTRO:-humble}/setup.bash
[ -f /home/ros/install/setup.bash ] && source /home/ros/install/setup.bash
echo "tracce: $CSV  robot: $ROBOT  predittore: $MODEL  ->  $OUT"
python3 -m pacer_core.dcal_from_tracks "$CSV" "$MODEL" "$OUT" --robot-id "$ROBOT" --n-mpc 6 --d-near 2.5
echo "fatto. Avvia con: ros2 launch pacer_bringup pacer_bringup.launch.py robot_body:=$ROBOT calibration:=$OUT"
