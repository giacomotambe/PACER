"""Filtro di sicurezza sul comando di velocita' (rev. 2026-11-17, nota PACER §3q-§3r).

Strato deterministico, indipendente da predittore e calibrazione: sta fra il controller Nav2
(PacerController, che pubblica su cmd_in_topic) e la base del robot (cmd_out_topic). Per ogni
comando controlla con pacer_core.safety_layer.unsafe_people che il robot possa ancora
fermarsi prima di toccare una persona (passive safety); se no, frena con a_dec.

Fail-safe: senza posa del robot (TF) o con tracce delle persone piu' vecchie di
tracks_timeout_s, frena. Le velocita' delle persone sono stimate dagli ultimi due punti dello
storico (cadenza dt_out del bridge).

Nav2: rimappare l'uscita del controller_server su cmd_in_topic (es. /cmd_vel_nav), e far
leggere alla base cmd_out_topic. Con il velocity_smoother di Nav2, questo nodo va DOPO lo
smoother (ultimo anello prima della base).
"""
import csv
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from pacer_msgs.msg import PedestrianTrackArray

from pacer_core.safety_layer import unsafe_people, brake_command


class SafetyFilterNode(Node):
    def __init__(self):
        super().__init__("pacer_safety_filter")
        dp = self.declare_parameter
        dp("cmd_in_topic", "/cmd_vel_nav")
        dp("cmd_out_topic", "/cmd_vel")
        dp("tracks_topic", "/pacer/pedestrian_tracks")
        dp("global_frame", "map")
        dp("base_frame", "base_link")
        dp("dt_tracks", 0.4)          # cadenza dello storico delle tracce [s]
        dp("t_cmd", 0.4)              # durata di un comando = periodo del controller [s]
        dp("a_dec", 1.5)              # [m/s^2] decelerazione garantita del robot
        dp("a_ped", 0.0)              # [m/s^2] deviazione delle persone (0 = velocita' costante)
        dp("clearance_l", 0.6)        # raggio robot + corpo persona [m]
        dp("margin", 0.1)             # [m]
        dp("tracks_timeout_s", 0.6)
        dp("enabled", True)
        dp("away_only", False)        # rev. 2026-11-22: stop solo se il moto avvicina la persona; rotazione a robot fermo
        dp("log_csv", "")             # un record per comando (anche quelli lasciati passare)
        g = lambda n: self.get_parameter(n).value
        self.p = {n: g(n) for n in ("dt_tracks", "t_cmd", "a_dec", "a_ped", "clearance_l", "margin",
                                    "tracks_timeout_s", "global_frame", "base_frame", "enabled", "away_only")}

        from tf2_ros import Buffer, TransformListener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(Twist, g("cmd_out_topic"), 10)
        self.create_subscription(Twist, g("cmd_in_topic"), self._on_cmd, 10)
        self.create_subscription(PedestrianTrackArray, g("tracks_topic"), self._on_tracks, 10)
        self.ped_pos = np.zeros((0, 2))
        self.ped_vel = np.zeros((0, 2))
        self.ped_ids = []
        self.t_tracks = None
        self.v_last, self.t_last = 0.0, None
        self.braking = False
        self.n_events = 0
        self._log = None
        if g("log_csv"):
            self._log_f = open(g("log_csv"), "a", newline="")
            self._log = csv.writer(self._log_f)
            if self._log_f.tell() == 0:
                self._log.writerow(["stamp", "v_in", "w_in", "v_out", "w_out", "action", "reason",
                                    "robot_x", "robot_y", "robot_th", "n_peds", "ids"])
        self.get_logger().info(
            f"pacer_safety_filter: {g('cmd_in_topic')} -> {g('cmd_out_topic')}, a_dec={self.p['a_dec']}, "
            f"a_ped={self.p['a_ped']}, enabled={self.p['enabled']}")

    def _on_tracks(self, msg):
        pos, vel, ids = [], [], []
        dt = float(self.p["dt_tracks"])
        for tr in msg.tracks:
            if not tr.history:
                continue
            p1 = np.array([tr.history[-1].x, tr.history[-1].y])
            if len(tr.history) >= 2:
                p0 = np.array([tr.history[-2].x, tr.history[-2].y])
                v = (p1 - p0) / dt
            else:
                v = np.zeros(2)       # appena entrato: velocita' ignota (coperta da margin)
            pos.append(p1); vel.append(v); ids.append(tr.id)
        self.ped_pos = np.array(pos).reshape(-1, 2)
        self.ped_vel = np.array(vel).reshape(-1, 2)
        self.ped_ids = ids
        self.t_tracks = time.monotonic()

    def _robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.p["global_frame"], self.p["base_frame"],
                                                 rclpy.time.Time())
        except Exception:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        th = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return np.array([t.x, t.y, th])

    def _on_cmd(self, cmd):
        now = time.monotonic()
        dt_since = (now - self.t_last) if self.t_last is not None else float(self.p["t_cmd"])
        v_in, w_in = float(cmd.linear.x), float(cmd.angular.z)
        x = self._robot_pose()
        action, reason, ids = "pass", "", []
        v_out, w_out = v_in, w_in
        if not self.p["enabled"]:
            reason = "disabled"
        elif x is None:
            action, reason = "brake", "no_tf"
        elif self.t_tracks is None or now - self.t_tracks > float(self.p["tracks_timeout_s"]):
            action, reason = "brake", "stale_tracks"
        else:
            idx = unsafe_people(x, max(v_in, 0.0), w_in, self.ped_pos, self.ped_vel,
                                t_cmd=float(self.p["t_cmd"]), a_dec=float(self.p["a_dec"]),
                                a_ped=float(self.p["a_ped"]), clearance_l=float(self.p["clearance_l"]),
                                margin=float(self.p["margin"]), away_only=bool(self.p["away_only"]))
            if idx:
                action, reason = "brake", "unsafe"
                ids = [self.ped_ids[i] for i in idx]
        if action == "brake":
            v_out, w_out = brake_command(self.v_last, dt_since, float(self.p["a_dec"]), w_in,
                                         rotate_when_stopped=bool(self.p["away_only"]) and reason == "unsafe")
            v_out = min(v_out, max(v_in, 0.0)) if v_in >= 0 else v_out
            if not self.braking:
                self.n_events += 1
                self.get_logger().warn(f"stop d'emergenza #{self.n_events}: {reason} {ids}")
        self.braking = action == "brake"
        out = Twist()
        out.linear.x, out.angular.z = float(v_out), float(w_out)
        self.pub.publish(out)
        self.v_last, self.t_last = float(v_out), now
        if self._log is not None:
            xs = x if x is not None else [float("nan")] * 3
            self._log.writerow([f"{self.get_clock().now().nanoseconds * 1e-9:.3f}", f"{v_in:.4f}",
                                f"{w_in:.4f}", f"{v_out:.4f}", f"{w_out:.4f}", action, reason,
                                f"{xs[0]:.4f}", f"{xs[1]:.4f}", f"{xs[2]:.4f}", len(self.ped_ids),
                                ";".join(ids)])
            self._log_f.flush()


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
