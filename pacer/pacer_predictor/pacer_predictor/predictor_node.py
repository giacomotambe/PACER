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
from geometry_msgs.msg import Point

from pacer_msgs.msg import (PedestrianTrackArray, PedestrianPrediction,
                              PedestrianPredictionArray)
from pacer_core.social_stgcnn import SocialSTGCNN
from pacer_core.predictor_utils import predict_window
from pacer_core.window_builder import build_window


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

        model_path = self.get_parameter("model_path").value
        self.obs_len = int(self.get_parameter("obs_len").value)
        self.pred_len = int(self.get_parameter("pred_len").value)
        self.max_neighbors = int(self.get_parameter("max_neighbors").value)

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

        self.get_logger().info("pacer_predictor avviato.")

    def _on_tracks(self, msg):
        histories = {t.id: np.array([[p.x, p.y] for p in t.history]) for t in msg.tracks}
        if not histories:
            return

        out = PedestrianPredictionArray()
        out.header = msg.header
        for pid in histories:
            window = build_window(histories, pid, self.obs_len, self.max_neighbors)
            if window is None:
                continue
            mu_abs, Sigma = predict_window(self.model, window)
            pred = PedestrianPrediction()
            pred.id = pid
            last_xy = histories[pid][-1]
            pred.current_xy = Point(x=float(last_xy[0]), y=float(last_xy[1]), z=0.0)
            pred.n_steps = mu_abs.shape[0]
            pred.mu = [Point(x=float(x), y=float(y), z=0.0) for x, y in mu_abs]
            pred.sigma_flat = Sigma.reshape(-1).astype(float).tolist()
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
