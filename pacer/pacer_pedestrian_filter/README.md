# pacer_pedestrian_filter

Toglie dalla point cloud del lidar i punti vicini ai pedoni tracciati,
**prima** che arrivi alla costmap di Nav2 — l'evitamento dei pedoni resta
così responsabilità esclusiva dell'NMPC (vincoli dedicati in `pacer_core`,
tramite `pacer_predictor`/`pacer_calibration`), invece di essere anche
marcato come ostacolo statico nella costmap.

## Perché

Senza questo nodo, `local_costmap`/`global_costmap` leggono il topic
lidar grezzo senza distinzioni: il corpo di un pedone nel campo visivo del
sensore viene marcato come ostacolo **statico**, mentre è già gestito come
ostacolo **dinamico** dai vincoli pedonali dell'NMPC. Due conseguenze
concrete osservate/previste:

1. Una scia di celle ad alto costo lungo tutta la traiettoria del pedone
   nella costmap, ripulita solo quando il lidar ripassa per raytracing su
   quelle celle.
2. `pacer_nav2_controller::PacerController::extractStaticObstacles()` (che
   legge le celle "lethal" della costmap senza sapere da dove vengono)
   passerebbe all'NMPC lo stesso pedone due volte: come vincolo statico
   *e* dinamico.

## Come funziona

Il nodo `pedestrian_filter_node`:

1. Si iscrive alla point cloud grezza del lidar (`input_topic`) e a
   `pacer_msgs/PedestrianTrackArray` (`tracks_topic`) — lo stesso
   messaggio/topic sia per `pacer_qualisys_bridge` (hardware reale) sia
   per `sim_pedestrian_publisher` (Gazebo): il nodo non distingue le due
   sorgenti.
2. Ad ogni point cloud in ingresso, trasforma via `tf2` l'ultima posizione
   nota di ciascun pedone nel frame della nuvola.
3. Rimuove (numpy vettorizzato) i punti entro `exclusion_radius` da
   ciascun pedone, ripubblica su `output_topic`.

Se non ci sono pedoni tracciati, o una trasformazione tf fallisce
(es. all'avvio, prima che il buffer tf si sia popolato), la point cloud
viene ripubblicata **invariata** quel ciclo — si preferisce non filtrare
piuttosto che bloccare/scartare dati del lidar.

## Parametri

| Parametro | Default | Note |
|---|---|---|
| `input_topic` | `/velodyne_points` | Point cloud grezza del lidar |
| `output_topic` | `/velodyne_points/pedestrian_filtered` | Da puntare nel `scan.topic` della costmap |
| `tracks_topic` | `/pacer/pedestrian_tracks` | `pacer_msgs/PedestrianTrackArray` |
| `exclusion_radius` | `0.6` (m) | Raggio attorno a ciascun pedone escluso dalla nuvola |
| `tf_timeout_s` | `0.1` | Timeout lookup tf pedone→frame nuvola |

Valori di progetto in `pacer_bringup/config/pacer_params.yaml` (sezione
`pacer_pedestrian_filter`, usata sia da `pacer_bringup.launch.py` che da
`pacer_sim`'s `pacer_gazebo_bringup.launch.py`).

## Integrazione con Nav2

`local_costmap`/`global_costmap` (in `pacer_sim/config/nav2_pacer_sim.yaml`
per la simulazione, o nel `nav2_params.yaml` del progetto per hardware
reale — vedi nota in `pacer_bringup/config/nav2_params_snippet.yaml`)
devono puntare `observation_sources.scan.topic` su `output_topic` di
questo nodo, **non** sul topic grezzo del lidar.
