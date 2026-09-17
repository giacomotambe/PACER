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
nota di ciascun pedone configurato, non ogni messaggio in arrivo."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from collections import deque
from geometry_msgs.msg import Point

from pacer_msgs.msg import PedestrianTrack, PedestrianTrackArray

try:
    from mocap_msgs.msg import RigidBodies
    _HAVE_MOCAP_MSGS = True
except ImportError:
    _HAVE_MOCAP_MSGS = False


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

        self.pedestrian_ids = [i for i in
                                 self.get_parameter("pedestrian_ids").value if i]
        self.obs_len = int(self.get_parameter("obs_len").value)
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
        self._history = {i: deque(maxlen=self.obs_len) for i in self.pedestrian_ids}

        if _HAVE_MOCAP_MSGS:
            self.create_subscription(RigidBodies, rb_topic, self._on_rigid_bodies,
                                       QoSPresetProfiles.SENSOR_DATA.value)

        self.pub = self.create_publisher(PedestrianTrackArray, out_topic, 10)
        self.timer = self.create_timer(self.dt_out, self._on_sample_tick)

        self.get_logger().info(
            f"pacer_qualisys_bridge avviato: {len(self.pedestrian_ids)} pedoni configurati, "
            f"campionamento a {1.0/self.dt_out:.2f} Hz (dt_out={self.dt_out}s), "
            f"obs_len={self.obs_len}.")

    def _on_rigid_bodies(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        for rb in msg.rigidbodies:
            name = rb.rigid_body_name
            if name not in self._history:
                continue   # non e' uno dei pedoni configurati (es. il robot stesso)
            p = rb.pose.pose.position if hasattr(rb.pose, "pose") else rb.pose.position
            self._latest_pose[name] = (float(p.x), float(p.y), now)

    def _on_sample_tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
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
            if len(hist) < self.obs_len:
                continue   # storico non ancora completo, non pubblicabile (§8:12)
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
