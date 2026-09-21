# pacer_sim

Ambiente di test Gazebo (gz-sim Harmonic) per `pacer-tupac-code-ros`: spawna
il robot Jo (da `jo_description`, vendored in questo repo), lo aggancia a
Nav2 (`jo_navigation`, `controller_server` puntato su `PacerController`) e
simula i pedoni al posto di Qualisys — non c'è motion capture in
simulazione, quindi al posto di `pacer_qualisys_bridge` gira
`sim_pedestrian_publisher`, che pubblica lo **stesso** messaggio
(`pacer_msgs/PedestrianTrackArray`) sullo stesso topic, con traiettorie
scriptate invece che marker reali — `pacer_predictor`/`pacer_calibration`/
`pacer_mpc_solver`/`pacer_nav2_controller` non cambiano di una riga.

## Uso

```bash
ros2 launch pacer_sim pacer_gazebo_bringup.launch.py
# con RViz:
ros2 launch pacer_sim pacer_gazebo_bringup.launch.py rviz:=true
```

Poi manda un goal (in frame `odom` — vedi "Perché niente mappa/AMCL" sotto),
per esempio:

```bash
ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "odom"}, pose: {position: {x: 8.0, y: 0.0}}}'
```

## Cosa fa il launch, in ordine

1. **`spawn_jo.launch.py`**: avvia `pacer_test_world.sdf` (pavimento +
   2 ostacoli statici) in gz-sim, pubblica `robot_state_publisher` da
   `urdf/jo_pacer_sim.urdf.xacro`, spawna il robot e fa da bridge gz↔ROS2
   per `/clock`, `/cmd_vel`, `/odom`, `/tf`, `/imu`, `/velodyne_points`.
2. **Nav2** (`jo_navigation/launch/navigation_local.launch.py`), modalità
   "local" (solo frame `odom`, niente AMCL/mappa — stessa modalità già
   usata da jo_navigation), con `config/nav2_pacer_sim.yaml`.
3. **`pacer_predictor` / `pacer_calibration` / `pacer_mpc_solver`**: gli
   stessi nodi usati con hardware reale, stessi parametri
   (`pacer_bringup/config/pacer_params.yaml`).
4. **`sim_pedestrian_publisher`**: sostituisce `pacer_qualisys_bridge` (vedi
   sopra).

## Differenze deliberate rispetto al robot/Nav2 reali di jo-zotac

- **Lidar**: il Velodyne reale di `jo_description` usa il plugin **RGL**
  (RobotecAI, GPU lidar) — una dipendenza extra non installata in questa
  immagine. `urdf/lidar_standard.xacro` lo sostituisce con un sensore
  `gpu_lidar` standard di gz-sim, sullo stesso punto di mount. Attenzione
  al topic: il sensore pubblica la `PointCloud2` su `<topic>/points`
  (`/velodyne_points/points`), non su `/velodyne_points` (quello porta un
  `LaserScan`) — il bridge in `spawn_jo.launch.py` la rimappa su
  `/velodyne_points` così `nav2_pacer_sim.yaml` resta identico a
  `jo_navigation/config/nav2_local.yaml` per quel pezzo.
- **Propulsione**: NON usa `gz-sim-tracked-vehicle-system` (i cingoli
  reali). Diagnosticato a fondo (log del plugin, test diretti su Ignition
  Transport bypassando ROS, riscontro nel riferimento `~/jo_sim_giacomo` —
  stesso problema lì, mai risolto): il robot si iscrive correttamente al
  topic ma non si muove mai, perché `TrackedVehicle` sostituisce la
  fisica di quelle collision con un proprio sistema di "contact surface
  customization" che in questa versione di gz-sim (**Fortress**/gz-sim 6,
  non Harmonic — è la coppia ufficiale con ROS2 Humble) non sembra produrre
  trazione reale. Sostituito con `gz-sim-diff-drive-system` standard su 2
  ruote motrici centrali + 2 supporti passivi a sfera (giunto fisso,
  attrito quasi nullo, solo per non far ribaltare uno chassis lungo
  0.94m) — tutti invisibili, staccati dalla geometria visiva dei cingoli
  (i cui giunti restano `fixed`, nessuna rotazione possibile). Una prima
  versione a 4 ruote (2 giunti per lato in `DiffDrive`) andava dritto ma
  non sterzava mai, con nessun'altra causa individuabile se non un
  limite/bug della configurazione multi-giunto-per-lato in questa
  versione del plugin — la configurazione a 2 ruote (la più
  diffusa/testata per `DiffDrive`) risolve.
- **Odometria**: sia quella interna di `TrackedVehicle` sia — per
  costruzione — quella di `DiffDrive` non vengono usate. Nav2 riceve
  invece odometria/TF dal plugin standard
  `gz-sim-odometry-publisher-system`, calcolata dalla posa reale del
  modello nel mondo simulato (ground truth) — più affidabile in
  simulazione, ma verifica comunque quanto diverge dal comportamento reale
  prima di fidartene per conclusioni sul robot vero.
- **Niente EKF/mappa/AMCL**: `jo_navigation/launch/localization.launch.py`
  si aspetta `glim_ros/odom_corrected` (SLAM lidar reale) e pubblicherebbe
  un secondo `odom→base_link` TF in conflitto con l'`OdometryPublisher` di
  cui sopra — escluso di proposito. Nav2 gira in modalità "local" (stessa
  di `jo_navigation`): niente `map_server`/AMCL, i goal vanno dati in frame
  `odom`.

## Stato: verificato end-to-end su hardware reale

`ros2 launch pacer_sim pacer_gazebo_bringup.launch.py rviz:=true` porta su
l'intero stack senza errori (GPU NVIDIA, macchina ibrida Intel+NVIDIA):
gz-sim + spawn di Jo, RViz, bridge gz↔ROS2, l'intero stack Nav2 attivo,
`PacerController` attivato, i 4 nodi pacer avviati, point cloud lidar
visibile in RViz. **Movimento confermato con la catena Nav2 completa**
(planner + `PacerController` + fisica): sia marcia dritta che sterzata
verificate con test diretti su `/odom` (non solo osservazione visiva).

Problemi risolti durante la verifica (vedi commenti nei file per i
dettagli — `nav2_pacer_sim.yaml`, `spawn_jo.launch.py`,
`urdf/jo_pacer_sim.urdf.xacro`):
- due plugin Nav2 (`nav2_navfn_planner`, `nav2_behaviors`) e l'ID del
  controller (`FollowPath`, non `PacerController` — è quello cablato nel
  behavior tree di default) con sintassi/nomi copiati da
  `jo_navigation`/`pacer_bringup` non riconosciuti dal pluginlib installato.
- rendering GPU: serve `/dev/dri` montato nel container e, su questa
  macchina ibrida Intel+NVIDIA, **non** forzare
  `__GLX_VENDOR_LIBRARY_NAME`/`__NV_PRIME_RENDER_OFFLOAD` (causa un crash
  NV-GLX) — lasciarli vuoti, come in jo-zotac (vedi `docker-compose.yml`).
- topic sbagliato nel bridge del lidar (`/velodyne_points` invece di
  `/velodyne_points/points`, vedi sopra).
- propulsione: `TrackedVehicle` → `DiffDrive` su 2 ruote + 2 caster, vedi
  sopra.

I `Timed out waiting for transform from base_link to odom` che compaiono
nei primissimi secondi di ogni avvio sono normali: il TF non è ancora
arrivato perché il robot è appena stato spawnato, si risolvono da soli in
~3s.

**Nota operativa**: se lanci/interrompi sessioni di test manualmente (non
solo `ros2 launch` interrotto con Ctrl+C e lasciato terminare da solo),
verifica che non restino processi orfani (`ps aux | grep -E 'ign|ros2'`)
prima di rilanciare — i nomi dei nodi (es. `robot_state_publisher`,
`predictor_node`) non contengono "ros2"/"ign"/"gz", quindi un `pkill -f`
su questi pattern non li tocca e possono restare due set di nodi paralleli
in conflitto sugli stessi topic (sintomo tipico: comportamento incoerente,
robot che sembra muoversi da solo).

## Risolto: overshoot sistematico sui goal (causa in pacer_core, corretto nel ponte Nav2)

Con un goal reale (es. `(6.0, 4.0)`, orientamento di default), il robot
si avvicina bene ma **supera ampiamente il goal senza fermarsi**, poi
inverte e torna indietro, oscillando senza mai convergere — verificato
con più giri di test puliti (log completo, traccia `/odom` nel tempo,
azione Nav2 seguita fino in fondo).

Due cause diverse individuate, non correlate ai problemi di
simulazione/fisica risolti sopra:

1. **`PacerController` non ragiona mai sull'orientamento del goal**:
   `pacer_msgs/srv/ComputeMpcCommand` porta solo `goal_x`/`goal_y`, mai
   un `goal_theta` — quindi anche a posizione raggiunta, se l'orientamento
   non combacia (tolleranza Nav2 di default 0.25 rad), il `goal_checker`
   non dichiara mai successo e il robot continua a rincorrere l'ultimo
   punto del path, ormai alle sue spalle. **Corretto** in
   `pacer_nav2_controller` (non in `pacer_core` — nessuna riscrittura del
   solver): quando il robot è entro `xy_goal_tolerance` dal goal finale,
   l'NMPC viene sostituito da una rotazione proporzionale pura
   (`xy_goal_tolerance`/`yaw_goal_tolerance`/`k_align`, nuovi parametri di
   `PacerController`, vedi `pacer_controller.hpp`/`.cpp` e
   `config/nav2_pacer_sim.yaml`).

2. **Causa principale — trovata e corretta**: instrumentando
   `PacerController` con un log temporaneo (robot pose, local goal, goal
   finale ad ogni ciclo), si è visto che il robot gira nei primi ~2 cicli
   di controllo, si stabilizza su un certo `θ`, e poi **avanza in linea
   perfettamente retta indefinitamente**, anche quando il bearing verso il
   proprio `local_goal` (dal punto di vista del robot, non dall'origine)
   richiederebbe una correzione di 0.35-0.45 rad. La causa è nella
   funzione di costo dell'NMPC in `pacer_core`
   (`unicycle_nmpc.py`/`unicycle_nmpc_soft.py`): penalizza **solo la
   distanza di posizione** dal goal ad ogni passo dell'orizzonte
   (`w_goal`, `w_stage`), mai l'orientamento/bearing verso il goal. Su un
   orizzonte breve (N_mpc=6, 2.4s) con warm-start, l'ottimizzatore trova
   una direzione "abbastanza buona" nel breve periodo e non ha una spinta
   esplicita a raffinarla — l'errore di rotta si congela e si accumula
   linearmente, causando l'overshoot sistematico osservato.

   **Corretto in `pacer_nav2_controller` (non in `pacer_core`)**: nuovo
   parametro `k_heading_bias` — dopo la risposta dell'NMPC, si somma
   all'`omega` restituito una correzione proporzionale all'errore di
   bearing verso `local_goal` (`v` e l'evitamento pedoni/statici restano
   interamente dell'NMPC, invariati). Completa il solver, non lo
   sostituisce.

   **Verificato**: stesso goal `(6.0, 4.0)` usato per diagnosticare il
   problema, traiettoria ora monotona verso il goal
   (`(0.49,0.19)→(3.11,2.04)→(4.34,2.93)→(5.63,3.76)→(5.98,3.99)`), si
   ferma a ~3cm dal goal e resta stabile; l'azione Nav2 riporta
   `Goal finished with status: SUCCEEDED` — non era mai successo prima di
   questo fix, in nessun test di questa sessione.

## Implementato: pedoni esclusi dalla costmap Nav2 (nuovo pacchetto `pacer_pedestrian_filter`)

Prima di questo fix, nessun meccanismo impediva ai pedoni tracciati (reali
via Qualisys, o simulati via `sim_pedestrian_publisher`) di finire anche
nella costmap di Nav2: `local_costmap`/`global_costmap` leggevano il topic
lidar grezzo (`/velodyne_points`) senza distinzioni, quindi il corpo di un
pedone nel campo visivo del sensore veniva marcato come ostacolo
**statico**, oltre a essere già gestito come ostacolo **dinamico** dai
vincoli pedonali dell'NMPC (`pacer_core`, via `pacer_predictor`/
`pacer_calibration`). Due problemi concreti: (1) una scia di celle ad alto
costo lungo tutta la traiettoria del pedone, che la costmap ripulisce solo
quando il lidar ripassa per raytracing su quelle celle; (2)
`PacerController::extractStaticObstacles()` (legge le celle "lethal" della
costmap senza sapere se provengono da un pedone) passava all'NMPC lo
stesso pedone due volte, come vincolo statico *e* dinamico.

**Corretto** con un nuovo pacchetto, `pacer_pedestrian_filter`
(`pacer/pacer_pedestrian_filter/`): il nodo `pedestrian_filter_node`
si iscrive alla point cloud grezza del lidar e a
`pacer_msgs/PedestrianTrackArray` (`/pacer/pedestrian_tracks` — stesso
topic/messaggio sia per Qualisys reale sia per `sim_pedestrian_publisher`,
il nodo non distingue le due sorgenti), trasforma via `tf2` l'ultima
posizione nota di ciascun pedone nel frame della nuvola, e ripubblica la
point cloud **senza** i punti entro `exclusion_radius` (default 0.6m) da
un pedone, su `/velodyne_points/pedestrian_filtered`. Sia
`config/nav2_pacer_sim.yaml` (qui) sia lo snippet
`pacer_bringup/config/nav2_params_snippet.yaml` (hardware reale) puntano
ora `local_costmap`/`global_costmap`'s `scan.topic` su questo topic
filtrato invece del `/velodyne_points` grezzo. L'evitamento dinamico resta
quindi interamente responsabilità dell'NMPC (vincoli pedonali di
`pacer_core`), la costmap vede solo ostacoli genuinamente statici.

Avviato automaticamente da `pacer_gazebo_bringup.launch.py` qui in
simulazione e da `pacer_bringup.launch.py` su hardware reale (parametri in
`pacer_bringup/config/pacer_params.yaml`, sezione
`pacer_pedestrian_filter`).

**Nota per il test in questa simulazione**: i pedoni di
`sim_pedestrian_publisher` sono traiettorie scriptate pubblicate solo su
`/pacer/pedestrian_tracks` — non sono entità fisiche in gz-sim, quindi il
lidar simulato non li "vede" mai e non c'è nulla da filtrare nella point
cloud in questo mondo di test. Il nodo va comunque verificato per
passthrough corretto (nessun errore, cloud ripubblicata invariata quando
non ci sono punti in raggio) — il filtro vero e proprio si osserva solo
con un pedone fisicamente presente nel mondo Gazebo o su hardware reale
con Qualisys.
