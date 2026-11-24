"""Inferenza online del predittore Social-STGCNN — legge
pacer_msgs/PedestrianTrackArray (oggi da pacer_qualisys_bridge, in futuro da
un vero tracker con la stessa interfaccia, §Sim2Real) e pubblica
pacer_msgs/PedestrianPredictionArray (mu, Sigma per pedone tracciato,
n_steps=H passi). UN SOLO forward pass per pedone predicibile, stesso
codice pacer_core.predictor_utils.predict_window gia' validato in
simulazione — questo nodo e' solo il ponte ROS2, nessuna logica nuova."""
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped

from pacer_msgs.msg import (PedestrianTrackArray, PedestrianPrediction,
                              PedestrianPredictionArray)
from pacer_core.social_stgcnn import SocialSTGCNN
from pacer_core.predictor_utils import predict_window, set_sigma_pow
from pacer_core.window_builder import build_window
from pacer_core.unpredicted import unpredicted_constraint, KIND_MODEL, CV_Q_SFM
from pacer_core.mondrian import assign_group, n_groups


class PredictorNode(Node):
    def __init__(self):
        super().__init__("pacer_predictor")

        self.declare_parameter("model_path", "")
        self.declare_parameter("obs_len", 8)
        self.declare_parameter("pred_len", 12)
        self.declare_parameter("hidden_dim", 32)
        self.declare_parameter("input_topic", "/pacer/pedestrian_tracks")
        self.declare_parameter("output_topic", "/pacer/pedestrian_predictions")
        self.declare_parameter("max_neighbors", 12)
        # rev. 2026-11-11: pedoni con storico corto (< obs_len), vedi pacer_core/unpredicted.py
        self.declare_parameter("unpred_mode", "cv")            # cv | cv1 | reach
        self.declare_parameter("cv_q", list(CV_Q_SFM))          # [m] raggi per k=1..n (quantile 0.999, SFM)
        self.declare_parameter("v_ped_cap", 2.5)                # [m/s] per il disco di raggiungibilita'
        self.declare_parameter("dt_out", 0.4)
        # rev. 2026-11-13: gruppo Mondrian assegnato qui, una volta, all'emissione
        # (pacer_core/mondrian.py). "none" | "dist" (distanza dal robot < d_near)
        self.declare_parameter("mondrian_group", "none")
        self.declare_parameter("d_near", 2.5)
        self.declare_parameter("robot_pose_topic", "/pacer/robot_pose")
        self.declare_parameter("robot_pose_timeout_s", 1.0)
        # rev. 2026-11-22 (nota §3s): Sigma^g, stesso valore di pacer_calibration e del D_cal
        self.declare_parameter("sigma_pow", 1.0)
        self.declare_parameter("sigma_pow_from_k", 1)   # 2 = k=1 invariato (rev. 2026-11-22, v45)
        set_sigma_pow(self.get_parameter("sigma_pow").value, self.get_parameter("sigma_pow_from_k").value)

        model_path = self.get_parameter("model_path").value
        self.obs_len = int(self.get_parameter("obs_len").value)
        self.pred_len = int(self.get_parameter("pred_len").value)
        self.max_neighbors = int(self.get_parameter("max_neighbors").value)
        self.unpred_mode = str(self.get_parameter("unpred_mode").value)
        self.cv_q = [float(v) for v in self.get_parameter("cv_q").value]
        self.v_ped_cap = float(self.get_parameter("v_ped_cap").value)
        self.dt_out = float(self.get_parameter("dt_out").value)
        self.mondrian_group = str(self.get_parameter("mondrian_group").value)
        n_groups(self.mondrian_group)                    # valida il parametro
        self.d_near = float(self.get_parameter("d_near").value)
        self.robot_pose_timeout_s = float(self.get_parameter("robot_pose_timeout_s").value)
        self._robot = None                               # (x, y, t_ricezione [s])

        self.model = SocialSTGCNN(obs_len=self.obs_len, pred_len=self.pred_len,
                                    hidden_dim=int(self.get_parameter("hidden_dim").value))
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.get_logger().info(f"pesi predittore caricati da {model_path}")
        else:
            self.get_logger().warn(
                "parametro 'model_path' vuoto: predittore con pesi NON allenati "
                "(inizializzazione casuale) — usare solo per test dell'integrazione, "
                "non per esperimenti. Allena/richiama il servizio TrainPredictor "
                "prima di un uso reale (§Sim2Real: codice di training richiamabile "
                "su richiesta).")
        self.model.eval()

        self.sub = self.create_subscription(
            PedestrianTrackArray, self.get_parameter("input_topic").value, self._on_tracks, 10)
        self.pub = self.create_publisher(
            PedestrianPredictionArray, self.get_parameter("output_topic").value, 10)
        self.create_subscription(PointStamped, self.get_parameter("robot_pose_topic").value,
                                 self._on_robot, 10)
        self._warned_no_robot = False

        self.get_logger().info("pacer_predictor avviato.")

    def _on_robot(self, msg):
        self._robot = (msg.point.x, msg.point.y, self.get_clock().now().nanoseconds * 1e-9)

    def _robot_xy(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._robot is None or now - self._robot[2] > self.robot_pose_timeout_s:
            if self.mondrian_group != "none" and not self._warned_no_robot:
                self.get_logger().warn("mondrian_group='dist' ma posa del robot assente o vecchia: "
                                       "pedoni con gruppo ignoto (MPC usa il quantile massimo, "
                                       "calibrazione li scarta). Controlla robot_body_name nel bridge.")
                self._warned_no_robot = True
            return None
        self._warned_no_robot = False
        return self._robot[:2]

    def _set_group(self, pred, robot_xy):
        g, d = assign_group(self.mondrian_group, (pred.current_xy.x, pred.current_xy.y),
                            robot_xy, self.d_near)
        pred.group = int(g)
        pred.d_robot = float(d)

    def _on_tracks(self, msg):
        all_hist = {t.id: np.array([[p.x, p.y] for p in t.history]).reshape(-1, 2)
                    for t in msg.tracks if t.history}
        if not all_hist:
            return
        # il modello vede solo gli storici completi (anche come vicini);
        # gli altri ricevono il vincolo senza predittore (rev. 2026-11-11)
        histories = {i: h for i, h in all_hist.items() if len(h) == self.obs_len}
        partial = {i: h for i, h in all_hist.items() if len(h) < self.obs_len}

        out = PedestrianPredictionArray()
        out.header = msg.header
        robot_xy = self._robot_xy()
        for pid, h in partial.items():
            kind, mu, radius = unpredicted_constraint(h, len(self.cv_q), self.unpred_mode, self.cv_q,
                                                      self.v_ped_cap, self.dt_out)
            pred = PedestrianPrediction()
            pred.id = pid
            pred.kind = kind
            pred.current_xy = Point(x=float(h[-1, 0]), y=float(h[-1, 1]), z=0.0)
            pred.n_steps = mu.shape[0]
            pred.mu = [Point(x=float(x), y=float(y), z=0.0) for x, y in mu]
            pred.sigma_flat = np.tile(np.eye(2).reshape(-1), mu.shape[0]).astype(float).tolist()
            pred.radius = radius.astype(float).tolist()
            self._set_group(pred, robot_xy)
            out.predictions.append(pred)
        for pid in histories:
            window = build_window(histories, pid, self.obs_len, self.max_neighbors)
            if window is None:
                continue
            mu_abs, Sigma = predict_window(self.model, window)
            pred = PedestrianPrediction()
            pred.id = pid
            pred.kind = KIND_MODEL
            last_xy = histories[pid][-1]
            pred.current_xy = Point(x=float(last_xy[0]), y=float(last_xy[1]), z=0.0)
            pred.n_steps = mu_abs.shape[0]
            pred.mu = [Point(x=float(x), y=float(y), z=0.0) for x, y in mu_abs]
            pred.sigma_flat = Sigma.reshape(-1).astype(float).tolist()
            self._set_group(pred, robot_xy)
            out.predictions.append(pred)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PredictorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
