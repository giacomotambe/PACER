# pacer_pedestrian_filter

Toglie dai dati del lidar i punti vicini ai pedoni tracciati,
**prima** che arrivino alla costmap di Nav2. Funziona con un lidar **2D**
(`LaserScan`, default, il lidar della macchinina) o **3D** (`PointCloud2`) — l'evitamento dei pedoni resta
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

1. Si iscrive al topic grezzo del lidar (`input_topic`) e a
   `pacer_msgs/PedestrianTrackArray` (`tracks_topic`) — lo stesso
   messaggio/topic pubblicato da `pacer_qualisys_bridge`.
2. Ad ogni messaggio del lidar, trasforma via `tf2` l'ultima posizione
   nota di ciascun pedone nel frame del lidar.
3. Ripubblica su `output_topic`:
   - **`input_type: scan`** (lidar 2D): i raggi il cui punto d'impatto e' entro
     `exclusion_radius` da un pedone vengono messi a `+inf` (`filtered_value`).
     Con `inf_is_valid: true` nell'obstacle layer, la costmap non marca il
     pedone e ripulisce per raytracing la scia che lascia. Gli altri raggi, e
     quelli gia' senza ritorno, restano identici: un muro davanti a un
     pedone nascosto non viene toccato.
   - **`input_type: cloud`** (lidar 3D): rimuove (numpy vettorizzato) i punti
     entro `exclusion_radius` da ciascun pedone.

Se non ci sono pedoni tracciati, o una trasformazione tf fallisce
(es. all'avvio, prima che il buffer tf si sia popolato), il messaggio del lidar
viene ripubblicato **invariato** quel ciclo — si preferisce non filtrare
piuttosto che bloccare/scartare dati del lidar.

## Parametri

| Parametro | Default | Note |
|---|---|---|
| `input_type` | `scan` | `scan` = lidar 2D (`LaserScan`), `cloud` = lidar 3D (`PointCloud2`) |
| `input_topic` | `/scan` | Topic grezzo del lidar (3D: es. `/velodyne_points`) |
| `output_topic` | `/scan/pedestrian_filtered` | Da puntare nel `scan.topic` della costmap |
| `filtered_value` | `inf` | Solo `scan`: valore dei raggi sui pedoni (`inf` o `nan`) |
| `tracks_topic` | `/pacer/pedestrian_tracks` | `pacer_msgs/PedestrianTrackArray` |
| `exclusion_radius` | `0.6` (m) | Raggio attorno a ciascun pedone escluso dalla nuvola |
| `tf_timeout_s` | `0.1` | Timeout lookup tf pedone→frame nuvola |

Valori di progetto in `pacer_bringup/config/pacer_params.yaml` (sezione
`pacer_pedestrian_filter`, usata da `pacer_bringup.launch.py`).

## Integrazione con Nav2

`local_costmap`/`global_costmap` (nel `nav2_params.yaml` del robot — vedi
nota in `pacer_bringup/config/nav2_params_snippet.yaml`)
devono puntare `observation_sources.scan.topic` su `output_topic` di
questo nodo, **non** sul topic grezzo del lidar. Con il lidar 2D:
`data_type: "LaserScan"` e `inf_is_valid: true` (esempio completo nello snippet).

Requisito tf: il frame dei pedoni (`world_frame` del bridge, di default `map`,
cioe' il frame di Qualisys) deve essere collegato al frame del lidar nell'albero tf.
