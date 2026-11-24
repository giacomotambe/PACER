# RUNBOOK — PACER sul NUC della macchinina

Procedura passo-passo per far girare PACER sulla macchinina reale. Il NUC non ha
GPU NVIDIA; Qualisys arriva via WiFi; Nav2 usa un lidar 2D. Le parti che dipendono
dal vostro hardware (driver del lidar, bringup Nav2, teleoperazione) sono tra `< >`
perché non sono in questo repository.

Flusso dei comandi: Nav2 → `/cmd_vel_nav` → filtro di sicurezza → `/cmd_vel` → base.
Localizzazione: Qualisys (il bridge pubblica `map → odom`), niente AMCL o SLAM.

---

## Fase 0 — Una volta sola: preparare il NUC

```bash
# Docker + plugin compose (se mancano)
sudo apt install docker.io docker-compose-v2 && sudo usermod -aG docker $USER   # poi logout/login

git clone https://github.com/giacomotambe/PACER.git && cd PACER
docker compose -f docker-compose.nuc.yml build        # 15-30 min la prima volta
```

**Rete WiFi.** Collega il NUC alla stessa rete WiFi del PC con QTM e controlla che
la porta 22222 sia raggiungibile:

```bash
timeout 3 bash -c 'echo > /dev/tcp/<IP_PC_QTM>/22222' && echo aperta
```

**In QTM:**
- la macchinina e ogni caschetto devono essere definiti come corpi 6DOF;
- lo streaming real-time deve essere attivo;
- segnati i nomi esatti dei corpi (maiuscole comprese).

## Fase 1 — Una volta sola: configurazione

I file sono nella cartella `PACER/` sull'host: il container li monta, quindi
non serve ricostruire l'immagine.

| File | Cosa impostare |
|---|---|
| `mocap4ros2_qualisys/qualisys_driver/config/qualisys_driver_params.yaml` | `host_name`: l'IP del PC QTM. `stream_frequency` resta 50 |
| `pacer/pacer_bringup/config/pacer_params.yaml` | `pedestrian_ids`: i nomi dei caschetti in QTM |
| stesso file | `body_to_base`: lo spostamento fra il rigid body e `base_link`. Resta `[0,0,0]` se in QTM hai messo l'origine al centro della macchinina con x in avanti |
| stesso file | `odom_frame`: `"odom"` se la macchinina pubblica l'odometria, `""` se non ce l'ha |
| stesso file | i limiti reali: `v2_a_acc`, `v2_a_dec` e `a_dec` del filtro di sicurezza (frenata garantita) |
| stesso file | `input_topic` del filtro pedoni: il topic del vostro lidar 2D, se non è `/scan` |
| il **vostro** `nav2_params.yaml` | le costmap devono leggere `/scan/pedestrian_filtered` con `data_type: LaserScan` e `inf_is_valid: true`, senza `static_layer` (rolling window). L'esempio è in `pacer/pacer_bringup/config/nav2_params_snippet.yaml` |
| il **vostro** bringup Nav2 | **niente AMCL o SLAM**. Il `controller_server` deve pubblicare su `/cmd_vel_nav` (remap `cmd_vel → cmd_vel_nav`) |

Il calibratore si sceglie in `pacer/pacer_bringup/config/calibrator.yaml`
(default `standard`) oppure al lancio con `calib_preset:=...`.

## Fase 2 — A ogni sessione: avvio

```bash
cd PACER
docker compose -f docker-compose.nuc.yml up -d      # al primo avvio compila il workspace (alcuni minuti)
```

Ogni shell qui sotto si apre con `docker compose -f docker-compose.nuc.yml exec pacer bash`.

| Shell | Comando |
|---|---|
| 1. Qualisys | `ros2 launch qualisys_driver qualisys.launch.py` |
| 2. Base + lidar | `<driver della base>` e `<driver del lidar 2D>`. Se girano fuori dal container, usa lo stesso `ROS_DOMAIN_ID` |
| 3. PACER | `ros2 launch pacer_bringup pacer_bringup.launch.py robot_body:=<nome_macchinina_QTM> run_name:=auto` |
| 4. Nav2 | `<vostro bringup Nav2>`, con il `controller_server` rimappato su `/cmd_vel_nav` |

Argomenti utili del launch di PACER:

| Argomento | Uso |
|---|---|
| `robot_body` | nome della macchinina in QTM |
| `run_name` | nome della prova per i log CSV; `auto` usa data e ora |
| `calib_preset` | `standard`, `conservative`, `legacy`, `static` o `custom` |
| `calibration` | file del D_cal (es. `/models/cal_real.npz`) |
| `model` | file del predittore |
| `pedestrian_filter` | `false` se non c'è lidar |

## Fase 3 — Controlli prima di muovere la macchinina (shell 5)

```bash
ros2 topic hz /rigid_bodies                 # ~50 Hz: Qualisys arriva via WiFi
ros2 run tf2_ros tf2_echo map base_link     # segue la macchinina come in QTM (localizzazione ok)
ros2 topic hz /pacer/pedestrian_tracks      # 2.5 Hz: pedoni tracciati
ros2 topic echo /pacer/q_tupac --once       # calibratore partito (preset standard, D_cal caricato)
ros2 topic hz /scan/pedestrian_filtered     # filtro del lidar attivo
python3 /home/ros/src/tools/mpc_jit_check.py   # tempi MPC sul NUC: se sono vicini a 250 ms, v2_jit: true
```

Se `tf2_echo` non trova la trasformata, la causa più probabile è un nome di
`robot_body` sbagliato o un'odometria assente. In quel caso imposta `odom_frame: ""`.

## Fase 4 — Prime prove: sicurezza prima di tutto

1. **Ruote sollevate:** manda un goal da RViz o con `ros2 action send_goal` e
   guarda i comandi su `/cmd_vel` senza che la macchinina si muova.
2. **A terra, a velocità ridotta:** abbassa temporaneamente la velocità massima
   nel vostro `nav2_params`, con una persona pronta all'arresto d'emergenza.
3. **Velocità piena.**

## Fase 5 — D_cal reale (necessario per gli esperimenti del paper)

Il `models/cal_sim.npz` incluso viene dalla simulazione e va bene per i primi test.
Per il paper serve quello reale: la garanzia presuppone che D_cal e prove vengano
dalle stesse condizioni.

1. Registra 10–20 minuti con la macchinina **in movimento fra le persone**
   (teleguidata con `<vostra teleop>` su `/cmd_vel_nav`, oppure in missione):
   ```bash
   ros2 launch pacer_bringup pacer_bringup.launch.py robot_body:=<nome> run_name:=dcal01
   ```
2. Costruisci il D_cal:
   ```bash
   /home/ros/src/tools/make_dcal_real.sh dcal01 <nome>      # -> /models/cal_real.npz (= PACER/models/ sull'host)
   ```
3. Da qui in poi lancia con il D_cal reale:
   ```bash
   ros2 launch pacer_bringup pacer_bringup.launch.py robot_body:=<nome> run_name:=auto calibration:=/models/cal_real.npz
   ```

## Fase 6 — Esperimenti

- Ogni prova con `run_name:=<nome_prova>` (o `auto`) salva quattro CSV in
  `PACER/bags/runs/<nome_prova>/`: `tracks.csv` (verità mocap di pedoni e robot),
  `scores.csv` (calibrazione), `mpc_status.csv` e `safety.csv`.
- Per i confronti fra calibratori usa `calib_preset:=standard`, `conservative`,
  `legacy` o `static`.
- Per registrare anche una bag delle pose:
  `/home/ros/src/mocap4ros2_qualisys/scripts/record_mocap.sh <prova> <nome> ped1 ped2`.

## Fase 7 — Fine sessione

```bash
docker compose -f docker-compose.nuc.yml down     # i dati restano in PACER/bags e PACER/models
```

---

## Se qualcosa non va

| Sintomo | Cosa fare |
|---|---|
| `/rigid_bodies` va a scatti o arriva in ritardo | abbassa `stream_frequency` a 20 in `qualisys_driver_params.yaml` e riavvia il driver |
| Nav2 si ferma con "transform timeout" | la posa di Qualisys è più vecchia di 0.3 s, per WiFi o occlusione: è voluto. Controlla la copertura delle telecamere |
| la macchinina frena senza motivo | guarda `safety.csv`: può essere un pedone a velocità costante che la intercetterebbe, oppure tracce di Qualisys più vecchie di 0.6 s |
| `tf2_echo map base_link` non trova la trasformata | controlla il nome in `robot_body`; senza odometria imposta `odom_frame: ""` |
| la costmap segna i pedoni come ostacoli | le costmap leggono il topic grezzo invece di `/scan/pedestrian_filtered`, oppure manca `inf_is_valid: true` |
| altri PC con ROS 2 sullo stesso WiFi | avvia con `ROS_DOMAIN_ID=17 docker compose -f docker-compose.nuc.yml up -d`, e lo stesso ID per i nodi fuori dal container |
| errore al build o alla compilazione | copia il messaggio d'errore: il build Docker e la compilazione C++ sul NUC non sono stati ancora provati |
