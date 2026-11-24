"""Toglie dalla point cloud del lidar i punti entro exclusion_radius da
ciascun pedone tracciato, PRIMA che arrivi alla costmap di Nav2.

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

Funziona identico con dati reali (Qualisys, via pacer_qualisys_bridge) o
simulati (sim_pedestrian_publisher): entrambi pubblicano lo stesso
pacer_msgs/PedestrianTrackArray sullo stesso topic — questo nodo non sa
(ne' deve sapere) da dove arrivano i pedoni.

I pedoni sono tracciati in world_frame (es. "map" — vedi
PedestrianTrackArray.header.frame_id), la
point cloud e' nel frame del sensore (es. "velodyne") — le posizioni dei
pedoni vengono trasformate nel frame della nuvola via tf2 ad ogni
messaggio."""
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401 — registra il tf2 buffer per PointStamped
import tf2_ros
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from pacer_msgs.msg import PedestrianTrackArray


class PedestrianPointCloudFilter(Node):
    def __init__(self):
        super().__init__("pacer_pedestrian_filter")

        self.declare_parameter("input_topic", "/velodyne_points")
        self.declare_parameter("output_topic", "/velodyne_points/pedestrian_filtered")
        self.declare_parameter("tracks_topic", "/pacer/pedestrian_tracks")
        self.declare_parameter("exclusion_radius", 0.6)
        self.declare_parameter("tf_timeout_s", 0.1)

        self.exclusion_radius = float(self.get_parameter("exclusion_radius").value)
        self.tf_timeout_s = float(self.get_parameter("tf_timeout_s").value)
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        tracks_topic = self.get_parameter("tracks_topic").value

        self._latest_tracks = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PedestrianTrackArray, tracks_topic, self._on_tracks, 10)
        self.create_subscription(
            PointCloud2, input_topic, self._on_cloud, qos_profile_sensor_data)
        self.pub = self.create_publisher(
            PointCloud2, output_topic, qos_profile_sensor_data)

        self.get_logger().info(
            f"pacer_pedestrian_filter avviato: {input_topic} -> {output_topic}, "
            f"pedoni da {tracks_topic}, raggio di esclusione {self.exclusion_radius}m.")

    def _on_tracks(self, msg):
        self._latest_tracks = msg

    def _exclusion_points_in_cloud_frame(self, cloud_header):
        """Ultima posizione nota di ciascun pedone, trasformata nel frame
        della point cloud. Lista vuota se non ci sono pedoni o la
        trasformazione fallisce (nessun filtro applicato quel ciclo,
        preferibile a bloccare/scartare la point cloud)."""
        if self._latest_tracks is None or not self._latest_tracks.tracks:
            return []

        points = []
        for track in self._latest_tracks.tracks:
            if not track.history:
                continue
            last = track.history[-1]
            ps = PointStamped()
            ps.header.frame_id = self._latest_tracks.header.frame_id
            ps.header.stamp = cloud_header.stamp
            ps.point = last
            try:
                transformed = self.tf_buffer.transform(
                    ps, cloud_header.frame_id,
                    timeout=Duration(seconds=self.tf_timeout_s))
                points.append((transformed.point.x, transformed.point.y))
            except Exception as exc:
                self.get_logger().warn(
                    f"trasformazione pedone '{track.id}' -> frame "
                    f"'{cloud_header.frame_id}' fallita: {exc}",
                    throttle_duration_sec=2.0)
        return points

    def _on_cloud(self, msg):
        exclusion_points = self._exclusion_points_in_cloud_frame(msg.header)
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


def main(args=None):
    rclpy.init(args=args)
    node = PedestrianPointCloudFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
