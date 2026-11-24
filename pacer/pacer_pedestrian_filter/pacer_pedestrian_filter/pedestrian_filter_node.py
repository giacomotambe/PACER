"""Toglie dai dati del lidar i punti entro exclusion_radius da ciascun pedone
tracciato, PRIMA che arrivino alla costmap di Nav2.

Due modalita' (parametro input_type):
  scan   lidar 2D, sensor_msgs/LaserScan (default dal 2026-11-24, il lidar della macchinina):
         i raggi che colpiscono un pedone vengono messi a +inf (filtered_value), cosi' la
         costmap non marca il pedone e, con inf_is_valid: true, ripulisce per raytracing
         la scia che il pedone lascia passando.
  cloud  lidar 3D, sensor_msgs/PointCloud2 (es. Velodyne): i punti vicini ai pedoni
         vengono tolti dalla nuvola.

Perche': pacer_nav2_controller::extractStaticObstacles() legge le celle
"lethal" della costmap SENZA distinzioni (vedi pacer_controller.cpp) — se
un pedone e' nel campo visivo del lidar, il suo corpo finisce anche li',
trattato come ostacolo STATICO oltre che DINAMICO (il costo/vincolo
pedonale dedicato di pacer_core, alimentato da
pacer_predictor/pacer_calibration). Risultato: doppio vincolo/chattering
(esattamente il rischio gia' descritto nel commento su "ambiguita'
statico/dinamico" in pacer_bringup/config/nav2_params_snippet.yaml) e una
scia di celle ad alto costo lungo tutta la traiettoria del pedone, che
persiste nella costmap (locale E globale, usata da NavFn per pianificare)
finche' il lidar non ripassa a "ripulire" quelle celle per raytracing.

I pedoni sono tracciati in world_frame (es. "map" — vedi
PedestrianTrackArray.header.frame_id), i dati del lidar sono nel frame del
sensore (es. "laser") — le posizioni dei pedoni vengono trasformate nel
frame del lidar via tf2 ad ogni messaggio."""
import math

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401 — registra il tf2 buffer per PointStamped
import tf2_ros
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from pacer_msgs.msg import PedestrianTrackArray


def scan_pedestrian_mask(ranges, angle_min, angle_increment, peds_xy, radius):
    """Maschera dei raggi di un LaserScan che terminano entro `radius` da almeno un pedone.

    ranges: array dei range [m] (inf/NaN = nessun ritorno, mai mascherati);
    peds_xy: lista di (x, y) dei pedoni nel frame del lidar. Funzione pura (testata offline).
    """
    r = np.asarray(ranges, dtype=np.float64)
    mask = np.zeros(r.shape, dtype=bool)
    if not len(peds_xy) or not r.size:
        return mask
    ang = angle_min + angle_increment * np.arange(r.size)
    valid = np.isfinite(r)
    x, y = r * np.cos(ang), r * np.sin(ang)
    r2 = radius * radius
    for px, py in peds_xy:
        mask |= valid & (((x - px) ** 2 + (y - py) ** 2) <= r2)
    return mask


class PedestrianFilter(Node):
    def __init__(self):
        super().__init__("pacer_pedestrian_filter")

        self.declare_parameter("input_type", "scan")                 # scan (lidar 2D) | cloud (lidar 3D)
        self.declare_parameter("input_topic", "/scan")
        self.declare_parameter("output_topic", "/scan/pedestrian_filtered")
        self.declare_parameter("tracks_topic", "/pacer/pedestrian_tracks")
        self.declare_parameter("exclusion_radius", 0.6)
        self.declare_parameter("tf_timeout_s", 0.1)
        self.declare_parameter("filtered_value", "inf")              # solo scan: inf | nan

        self.input_type = str(self.get_parameter("input_type").value)
        if self.input_type not in ("scan", "cloud"):
            raise ValueError(f"input_type sconosciuto: '{self.input_type}' (scan | cloud)")
        self.exclusion_radius = float(self.get_parameter("exclusion_radius").value)
        self.tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)
        fv = str(self.get_parameter("filtered_value").value)
        self.filtered_value = math.inf if fv == "inf" else math.nan
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        tracks_topic = self.get_parameter("tracks_topic").value

        self._latest_tracks = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PedestrianTrackArray, tracks_topic, self._on_tracks, 10)
        msg_type, cb = (LaserScan, self._on_scan) if self.input_type == "scan" else (PointCloud2, self._on_cloud)
        self.create_subscription(msg_type, input_topic, cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(msg_type, output_topic, qos_profile_sensor_data)

        self.get_logger().info(
            f"pacer_pedestrian_filter avviato ({self.input_type}): {input_topic} -> {output_topic}, "
            f"pedoni da {tracks_topic}, raggio di esclusione {self.exclusion_radius}m.")

    def _on_tracks(self, msg):
        self._latest_tracks = msg

    def _exclusion_points(self, header):
        """Ultima posizione nota di ciascun pedone, trasformata nel frame
        del lidar. Lista vuota se non ci sono pedoni o la trasformazione
        fallisce (nessun filtro applicato quel ciclo, preferibile a
        bloccare/scartare i dati del lidar)."""
        if self._latest_tracks is None or not self._latest_tracks.tracks:
            return []

        points = []
        for track in self._latest_tracks.tracks:
            if not track.history:
                continue
            last = track.history[-1]
            ps = PointStamped()
            ps.header.frame_id = self._latest_tracks.header.frame_id
            ps.header.stamp = header.stamp
            ps.point = last
            try:
                transformed = self.tf_buffer.transform(
                    ps, header.frame_id,
                    timeout=Duration(seconds=self.tf_timeout_s))
                points.append((transformed.point.x, transformed.point.y))
            except Exception as exc:
                self.get_logger().warn(
                    f"trasformazione pedone '{track.id}' -> frame "
                    f"'{header.frame_id}' fallita: {exc}",
                    throttle_duration_sec=2.0)
        return points

    def _on_scan(self, msg):
        peds = self._exclusion_points(msg.header)
        if not peds:
            self.pub.publish(msg)
            return
        mask = scan_pedestrian_mask(msg.ranges, msg.angle_min, msg.angle_increment,
                                    peds, self.exclusion_radius)
        if mask.any():
            ranges = np.asarray(msg.ranges, dtype=np.float32)
            ranges[mask] = self.filtered_value
            msg.ranges = ranges.tolist()
        self.pub.publish(msg)

    def _on_cloud(self, msg):
        exclusion_points = self._exclusion_points(msg.header)
        if not exclusion_points:
            self.pub.publish(msg)
            return

        # read_points ritorna un ndarray STRUTTURATO (campi nominati x/y/z,
        # non colonne semplici) — va convertito esplicitamente per campo,
        # un cast diretto dell'intero array a float32 fallisce (TypeError).
        cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        if cloud.size == 0:
            self.pub.publish(msg)
            return
        pts = np.column_stack(
            [cloud["x"], cloud["y"], cloud["z"]]).astype(np.float32)

        r2 = self.exclusion_radius ** 2
        keep_mask = np.ones(len(pts), dtype=bool)
        for ex, ey in exclusion_points:
            d2 = (pts[:, 0] - ex) ** 2 + (pts[:, 1] - ey) ** 2
            keep_mask &= (d2 > r2)

        filtered = pts[keep_mask]
        out = pc2.create_cloud_xyz32(msg.header, filtered.tolist())
        self.pub.publish(out)


# nome precedente, per chi lo importava
PedestrianPointCloudFilter = PedestrianFilter


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
