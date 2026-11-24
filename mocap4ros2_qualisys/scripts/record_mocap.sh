#!/bin/bash
# Registra una bag con la posa di uno o più rigid body (default: traquad)
# Uso: record_mocap.sh <nome_esperimento> [rigid_body ...]
# Da github.com/giacomotambe/qualisys_docker (4c2e2f1); nel container PACER e' in
# /home/ros/src/mocap4ros2_qualisys/scripts e le bag vanno in /home/ros/bags (= ./bags del repo).
set -e

if [ -z "$1" ]; then
    echo "Uso: $0 <nome_esperimento> [rigid_body ...]"
    exit 1
fi

EXPERIMENT="$1"
shift
BODIES=("$@")
if [ ${#BODIES[@]} -eq 0 ]; then
    BODIES=(traquad)
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BAG_DIR="$HOME/bags/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"

if [[ "$EXPERIMENT" == */* ]]; then
    echo "Il nome dell'esperimento non può contenere '/'"
    exit 1
fi

source /opt/ros/humble/setup.bash
# workspace del container PACER (rev. 2026-11-24): /home/ros, non /home/ros/mocap_ws come in qualisys_docker
source /home/ros/install/setup.bash

# Il driver deve essere già attivo
if ! ros2 topic list | grep -qx "/rigid_bodies"; then
    echo "/rigid_bodies non trovato: avvia prima 'ros2 launch qualisys_driver qualisys.launch.py'"
    exit 1
fi

python3 "$SCRIPT_DIR/rigid_body_filter.py" "${BODIES[@]}" &
FILTER_PID=$!
trap 'kill -INT $FILTER_PID 2>/dev/null; wait $FILTER_PID 2>/dev/null' EXIT

# Aspetta che arrivi almeno un messaggio da ogni rigid body
TOPICS=()
for BODY in "${BODIES[@]}"; do
    echo "In attesa di dati da '$BODY'..."
    if ! timeout 10 ros2 topic echo --once "/$BODY/pose" geometry_msgs/msg/PoseStamped > /dev/null; then
        echo "Nessun dato da '$BODY' in 10 s: controlla il nome del 6DOF body in QTM"
        exit 1
    fi
    TOPICS+=("/$BODY/pose")
done

echo "Registrazione di ${TOPICS[*]} in $BAG_DIR (Ctrl+C per fermare)"
ros2 bag record -o "$BAG_DIR" "${TOPICS[@]}"
