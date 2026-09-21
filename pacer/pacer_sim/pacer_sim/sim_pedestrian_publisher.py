"""Sostituto di pacer_qualisys_bridge per la simulazione Gazebo: non c'e'
motion capture in sim, quindi qui i pedoni non arrivano da marker Qualisys
ma da traiettorie scriptate (waypoint a andirivieni, velocita' costante).

Contratto di uscita IDENTICO a pacer_qualisys_bridge/qualisys_bridge_node.py
(stesso topic di default, stesso messaggio, stessa logica di
accumulo/pubblicazione: un timer a 1/dt_out Hz appende l'ultima posa nota di
ciascun pedone a una deque(maxlen=obs_len) e pubblica solo i pedoni con
storico gia' pieno) — cosi' pacer_predictor/pacer_calibration/
pacer_nav2_controller non cambiano di una riga passando da hardware reale a
simulazione: vedi pacer_gazebo_bringup.launch.py, che lancia questo nodo al
posto di pacer_qualisys_bridge.
"""
from collections import deque

import rclpy
from geometry_msgs.msg import Point
from pacer_msgs.msg import PedestrianTrack, PedestrianTrackArray
from rclpy.node import Node


class ScriptedPedestrian:
    """Cammina a velocita' costante lungo una spezzata, facendo
    andirivieni (torna indietro invece di teletrasportarsi al primo
    waypoint) cosi' resta nel raggio d'azione del robot per tutta la
    durata di un test, invece di allontanarsi in una direzione sola."""

    def __init__(self, waypoints_x, waypoints_y, speed):
        if len(waypoints_x) != len(waypoints_y) or len(waypoints_x) < 2:
            raise ValueError("waypoints_x/waypoints_y devono avere la stessa "
                              "lunghezza (>=2)")
        self.waypoints = list(zip(waypoints_x, waypoints_y))
        self.speed = max(1e-3, float(speed))
        self.x, self.y = self.waypoints[0]
        self._target_idx = 1
        self._direction = 1   # 1 = avanti lungo la lista, -1 = indietro

    def step(self, dt):
        tx, ty = self.waypoints[self._target_idx]
        dx, dy = tx - self.x, ty - self.y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        move = self.speed * dt
        if dist <= move or dist < 1e-6:
            self.x, self.y = tx, ty
            self._advance_target()
        else:
            self.x += dx / dist * move
            self.y += dy / dist * move

    def _advance_target(self):
        n = len(self.waypoints)
        nxt = self._target_idx + self._direction
        if nxt < 0 or nxt >= n:
            self._direction *= -1   # fine spezzata: inverti verso
            nxt = self._target_idx + self._direction
        self._target_idx = nxt


class SimPedestrianPublisher(Node):
    def __init__(self):
        super().__init__("sim_pedestrian_publisher")

        self.declare_parameter("pedestrian_ids", ["ped1", "ped2", "ped3"])
        self.declare_parameter("obs_len", 8)
        self.declare_parameter("dt_out", 0.4)
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("output_topic", "/pacer/pedestrian_tracks")
        self.declare_parameter("sim_rate_hz", 20.0)

        self.obs_len = int(self.get_parameter("obs_len").value)
        self.dt_out = float(self.get_parameter("dt_out").value)
        self.world_frame = self.get_parameter("world_frame").value
        out_topic = self.get_parameter("output_topic").value
        sim_rate_hz = float(self.get_parameter("sim_rate_hz").value)
        pedestrian_ids = [i for i in self.get_parameter("pedestrian_ids").value if i]

        self._pedestrians = {}
        for pid in pedestrian_ids:
            self.declare_parameter(f"{pid}.waypoints_x", [-3.0, 3.0])
            self.declare_parameter(f"{pid}.waypoints_y", [0.0, 0.0])
            self.declare_parameter(f"{pid}.speed", 1.0)
            wx = list(self.get_parameter(f"{pid}.waypoints_x").value)
            wy = list(self.get_parameter(f"{pid}.waypoints_y").value)
            speed = float(self.get_parameter(f"{pid}.speed").value)
            try:
                self._pedestrians[pid] = ScriptedPedestrian(wx, wy, speed)
            except ValueError as exc:
                self.get_logger().error(f"pedone '{pid}' scartato: {exc}")

        self._history = {pid: deque(maxlen=self.obs_len) for pid in self._pedestrians}

        self.pub = self.create_publisher(PedestrianTrackArray, out_topic, 10)
        self._motion_timer = self.create_timer(1.0 / sim_rate_hz, self._on_motion_tick)
        self._sample_timer = self.create_timer(self.dt_out, self._on_sample_tick)

        self.get_logger().info(
            f"sim_pedestrian_publisher avviato: {len(self._pedestrians)} pedoni "
            f"scriptati, campionamento a {1.0 / self.dt_out:.2f} Hz (dt_out="
            f"{self.dt_out}s), obs_len={self.obs_len}. Sostituisce "
            f"pacer_qualisys_bridge — nessuna motion capture in simulazione.")

    def _on_motion_tick(self):
        dt = 1.0 / float(self.get_parameter("sim_rate_hz").value)
        for ped in self._pedestrians.values():
            ped.step(dt)

    def _on_sample_tick(self):
        for pid, ped in self._pedestrians.items():
            self._history[pid].append((ped.x, ped.y))

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
    node = SimPedestrianPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
