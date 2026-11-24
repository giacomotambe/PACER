"""Ponte Qualisys -> pacer_msgs/PedestrianTrackArray (§Sim2Real "aggiornamento
— ground truth pedoni per la campagna sperimentale attuale"): pedoni con un
caschetto a markers riflettenti, tracciati da Qualisys con identita' SEMPRE
certa (nessun ID switching) — sostituisce per ora il modulo di
riconoscimento/tracking reale (camera/lidar), che restera' lavoro futuro
CON LA STESSA INTERFACCIA A VALLE (stesso topic/messaggio), cosi' che
pacer_predictor/pacer_calibration non debbano cambiare quando verra'
integrato.

Assunzione sul driver Qualisys: pubblica mocap_msgs/msg/RigidBodies su un
topic (default /rigid_bodies), convenzione dell'ecosistema mocap4ros2/
qualisys_ros2 (un mocap_msgs/msg/RigidBody per corpo tracciato, con
rigid_body_name e pose). Se il driver installato usa un pacchetto/messaggio
diverso, adattare SOLO _on_rigid_bodies() qui sotto — il resto del nodo
(bufferizzazione, resampling, pubblicazione) resta invariato.

Resampling: Qualisys tipicamente streamma a ~100+ Hz, il predittore Social-
STGCNN e' stato allenato con cadenza dt_out=0.4s (protocollo 8:12, stesso
della pipeline simulata) — un timer a 1/dt_out Hz campiona l'ULTIMA posa
nota di ciascun pedone configurato, non ogni messaggio in arrivo.

Localizzazione da Qualisys (rev. 2026-11-24): con robot_body_name impostato e
publish_localization_tf true, il nodo pubblica su /tf la trasformata
world_frame -> odom_frame (come farebbe AMCL), calcolata dalla posa mocap del robot e
dall'odometria della base (odom_frame -> base_frame, letta da tf):
    T_map_odom = T_map_body * T_body_base * inv(T_odom_base)
Cosi' Nav2, il filtro dei pedoni e l'MPC lavorano tutti nel frame di Qualisys, dove sono
pubblicati i pedoni. Con odom_frame "" pubblica direttamente world_frame -> base_frame
(robot senza odometria). Se la posa mocap del robot e' vecchia (> tf_max_age_s) non
pubblica nulla: Nav2 si ferma per tf scaduta invece di usare una posa sbagliata."""
import math

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from collections import deque
from geometry_msgs.msg import Point, PointStamped, TransformStamped

from pacer_msgs.msg import PedestrianTrack, PedestrianTrackArray

try:
    from mocap_msgs.msg import RigidBodies
    _HAVE_MOCAP_MSGS = True
except ImportError:
    _HAVE_MOCAP_MSGS = False


def yaw_from_quaternion(q):
    """Yaw (rotazione attorno a z) da un quaternione geometry_msgs (x, y, z, w)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def compose2d(a, b):
    """Pose planari (x, y, yaw): a * b."""
    ax, ay, at = a
    bx, by, bt = b
    c, s = math.cos(at), math.sin(at)
    return (ax + c * bx - s * by, ay + s * bx + c * by, math.atan2(math.sin(at + bt), math.cos(at + bt)))


def invert2d(a):
    ax, ay, at = a
    c, s = math.cos(at), math.sin(at)
    return (-(c * ax + s * ay), -(-s * ax + c * ay), -at)


def map_to_odom(map_body, body_base, odom_base):
    """T_map_odom = T_map_body * T_body_base * inv(T_odom_base), tutte pose planari (x, y, yaw)."""
    return compose2d(compose2d(map_body, body_base), invert2d(odom_base))


class QualisysBridgeNode(Node):
    def __init__(self):
        super().__init__("pacer_qualisys_bridge")

        self.declare_parameter("pedestrian_ids", [""])
        self.declare_parameter("rigid_bodies_topic", "/rigid_bodies")
        self.declare_parameter("output_topic", "/pacer/pedestrian_tracks")
        self.declare_parameter("obs_len", 8)
        self.declare_parameter("dt_out", 0.4)
        self.declare_parameter("occlusion_timeout_s", 1.0)
        self.declare_parameter("world_frame", "map")
        # rev. 2026-11-04: log CSV a dt_out di pedoni e robot (verita' mocap)
        # per D_cal reale (shadow mode) e per l'analisi offline delle prove
        self.declare_parameter("tracks_log_csv", "")
        self.declare_parameter("robot_body_name", "")
        # rev. 2026-11-11: pubblica anche gli storici incompleti (< obs_len).
        # Prima venivano scartati e il pedone era INVISIBILE all'MPC per i
        # primi obs_len*dt_out = 3.2 s dopo l'ingresso o un'occlusione.
        self.declare_parameter("publish_partial", True)
        # rev. 2026-11-13: posa del robot (rigid body robot_body_name) pubblicata a
        # dt_out, usata da pacer_predictor per il gruppo Mondrian per distanza
        self.declare_parameter("robot_pose_topic", "/pacer/robot_pose")
        # rev. 2026-11-24: localizzazione da Qualisys (vedi docstring del modulo)
        self.declare_parameter("publish_localization_tf", True)
        self.declare_parameter("odom_frame", "odom")        # "" = pubblica world_frame -> base_frame
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("body_to_base", [0.0, 0.0, 0.0])   # posa di base_frame nel rigid body [m, m, rad]
        self.declare_parameter("tf_rate", 30.0)             # [Hz]
        self.declare_parameter("tf_max_age_s", 0.3)         # posa mocap piu' vecchia -> nessuna tf
        self.declare_parameter("tf_post_date_s", 0.1)       # come transform_tolerance di AMCL

        self.pedestrian_ids = [i for i in
                                 self.get_parameter("pedestrian_ids").value if i]
        self.obs_len = int(self.get_parameter("obs_len").value)
        self.publish_partial = bool(self.get_parameter("publish_partial").value)
        self.dt_out = float(self.get_parameter("dt_out").value)
        self.occlusion_timeout_s = float(self.get_parameter("occlusion_timeout_s").value)
        self.world_frame = self.get_parameter("world_frame").value
        out_topic = self.get_parameter("output_topic").value
        rb_topic = self.get_parameter("rigid_bodies_topic").value

        if not self.pedestrian_ids:
            self.get_logger().warn(
                "parametro 'pedestrian_ids' vuoto: nessun pedone verra' tracciato. "
                "Configuralo con i nomi dei rigid body Qualisys dei caschetti.")
        if not _HAVE_MOCAP_MSGS:
            self.get_logger().error(
                "pacchetto 'mocap_msgs' non trovato — adatta _on_rigid_bodies() al "
                "messaggio effettivamente pubblicato dal tuo driver Qualisys.")

        self._latest_pose = {}      # id -> (x,y,stamp_sec)
        self._robot_name = self.get_parameter("robot_body_name").value
        self._tick = 0
        self._log = None
        log_path = self.get_parameter("tracks_log_csv").value
        if log_path:
            import csv
            self._log_f = open(log_path, "a", newline="")
            self._log = csv.writer(self._log_f)
            if self._log_f.tell() == 0:
                self._log.writerow(["tick", "stamp", "id", "x", "y", "fresh"])
        self._history = {i: deque(maxlen=self.obs_len) for i in self.pedestrian_ids}

        if _HAVE_MOCAP_MSGS:
            self.create_subscription(RigidBodies, rb_topic, self._on_rigid_bodies,
                                       QoSPresetProfiles.SENSOR_DATA.value)

        self.pub = self.create_publisher(PedestrianTrackArray, out_topic, 10)
        self.robot_pub = (self.create_publisher(PointStamped, self.get_parameter("robot_pose_topic").value, 10)
                          if self._robot_name else None)
        self.timer = self.create_timer(self.dt_out, self._on_sample_tick)

        self._loc_tf = bool(self.get_parameter("publish_localization_tf").value) and bool(self._robot_name)
        if self._loc_tf:
            import tf2_ros
            self.odom_frame = str(self.get_parameter("odom_frame").value)
            self.base_frame = str(self.get_parameter("base_frame").value)
            self.body_to_base = tuple(float(v) for v in self.get_parameter("body_to_base").value)
            self.tf_max_age_s = float(self.get_parameter("tf_max_age_s").value)
            self.tf_post_date_s = float(self.get_parameter("tf_post_date_s").value)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            if self.odom_frame:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.create_timer(1.0 / float(self.get_parameter("tf_rate").value), self._on_tf_tick)
            self.get_logger().info(
                f"localizzazione da Qualisys: {self.world_frame} -> "
                f"{self.odom_frame or self.base_frame} dal rigid body '{self._robot_name}'.")
        elif bool(self.get_parameter("publish_localization_tf").value):
            self.get_logger().warn("publish_localization_tf senza robot_body_name: nessuna tf pubblicata.")

        self.get_logger().info(
            f"pacer_qualisys_bridge avviato: {len(self.pedestrian_ids)} pedoni configurati, "
            f"campionamento a {1.0/self.dt_out:.2f} Hz (dt_out={self.dt_out}s), "
            f"obs_len={self.obs_len}.")

    def _on_rigid_bodies(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        for rb in msg.rigidbodies:
            name = rb.rigid_body_name
            if name not in self._history and name != self._robot_name:
                continue   # non e' uno dei pedoni configurati ne' il robot
            pose = rb.pose.pose if hasattr(rb.pose, "pose") else rb.pose
            p = pose.position
            self._latest_pose[name] = (float(p.x), float(p.y), now)
            if name == self._robot_name:
                self._robot_pose3 = (float(p.x), float(p.y), yaw_from_quaternion(pose.orientation), now)

    def _on_tf_tick(self):
        e = getattr(self, "_robot_pose3", None)
        now = self.get_clock().now()
        if e is None or now.nanoseconds * 1e-9 - e[3] > self.tf_max_age_s:
            self.get_logger().warn(
                f"posa Qualisys di '{self._robot_name}' assente o vecchia: nessuna tf di localizzazione",
                throttle_duration_sec=2.0)
            return
        map_base = compose2d(e[:3], self.body_to_base)
        if self.odom_frame:
            try:
                t = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, rclpy.time.Time())
            except Exception as exc:
                self.get_logger().warn(f"tf {self.odom_frame} -> {self.base_frame} non disponibile: {exc}",
                                       throttle_duration_sec=2.0)
                return
            tr = t.transform.translation
            odom_base = (tr.x, tr.y, yaw_from_quaternion(t.transform.rotation))
            pose, child = compose2d(map_base, invert2d(odom_base)), self.odom_frame
        else:
            pose, child = map_base, self.base_frame
        msg = TransformStamped()
        msg.header.stamp = (now + rclpy.duration.Duration(seconds=self.tf_post_date_s)).to_msg()
        msg.header.frame_id = self.world_frame
        msg.child_frame_id = child
        msg.transform.translation.x, msg.transform.translation.y = pose[0], pose[1]
        msg.transform.rotation.z, msg.transform.rotation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
        self.tf_broadcaster.sendTransform(msg)

    def _on_sample_tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._log is not None:
            for name in self.pedestrian_ids + ([self._robot_name] if self._robot_name else []):
                e = self._latest_pose.get(name)
                if e is not None:
                    fresh = int(now - e[2] <= self.occlusion_timeout_s)
                    self._log.writerow([self._tick, f"{now:.3f}", name, f"{e[0]:.4f}",
                                        f"{e[1]:.4f}", fresh])
            self._log_f.flush()
        self._tick += 1
        if self.robot_pub is not None:
            e = self._latest_pose.get(self._robot_name)
            if e is not None and now - e[2] <= self.occlusion_timeout_s:
                rp = PointStamped()
                rp.header.stamp = self.get_clock().now().to_msg()
                rp.header.frame_id = self.world_frame
                rp.point = Point(x=e[0], y=e[1], z=0.0)
                self.robot_pub.publish(rp)
        for pid in self.pedestrian_ids:
            entry = self._latest_pose.get(pid)
            if entry is None:
                continue
            x, y, t_seen = entry
            if now - t_seen > self.occlusion_timeout_s:
                # marker occluso/perso troppo a lungo: storico invalidato,
                # va ricostruito da zero (stessa politica di
                # LiveOpenField su rigenerazione, §Sim2Real criticita'
                # "continuita' del tracking" — qui non c'e' rischio di ID
                # switch [Qualisys], solo di buco nello storico).
                self._history[pid].clear()
                continue
            self._history[pid].append((x, y))

        msg = PedestrianTrackArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        for pid, hist in self._history.items():
            if not hist or (len(hist) < self.obs_len and not self.publish_partial):
                continue   # storico vuoto (o incompleto con publish_partial=false)
            track = PedestrianTrack()
            track.id = pid
            track.history = [Point(x=float(x), y=float(y), z=0.0) for x, y in hist]
            msg.tracks.append(track)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = QualisysBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
