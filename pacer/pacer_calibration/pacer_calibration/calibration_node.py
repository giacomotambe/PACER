"""Calibrazione TUPAC: online (§Sim2Real "il nodo che opera online... calcola
online q") + i due servizi ON-DEMAND (training del predittore, ricostruzione
di D_cal) che devono restare SEMPRE disponibili e mai nascosti nel nodo
online (§Sim2Real, requisito esplicito). Riusa pacer_core.calibration_worker
.OnlineCalibrationWorker (thread separato dal resto del nodo, stessa
architettura validata in simulazione: il ricalcolo O(s) del quantile non
blocca mai la lettura dell'ultimo q_tupac(k) disponibile) e
pacer_core.calibration_offline (build_dcal_from_sfm/quantile_warmstart, per
ora) per i due servizi.

Maturazione online: ad ogni PedestrianPredictionArray in arrivo si avanza un
contatore di ciclo t_now e si schedulano gli eventi a t_now+k (k=1..n_mpc,
stessa logica di schedule_calibration nella simulazione); alla ricezione
successiva si risolvono gli eventi schedulati per l'attuale t_now contro
l'ULTIMA posizione reale nota di quell'id (dalla cache aggiornata da
PedestrianTrackArray — Qualisys, identita' certa: nessun controllo di ID
switch necessario oggi, ma il campo e' gia' predisposto per quando servira',
§Sim2Real criticita' "continuita' del tracking")."""
import os
import time
from collections import defaultdict
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from pacer_msgs.msg import PedestrianPredictionArray, PedestrianTrackArray, QTupacSnapshot
from pacer_msgs.srv import TrainPredictor, RecalibrateTupac

from pacer_core.calibration_worker import OnlineCalibrationWorker
from pacer_core.nonconformity import mahalanobis_score
from pacer_core.quantile_tupac import HFunction
from pacer_core import calibration_offline as calib_off
from pacer_core.social_stgcnn import SocialSTGCNN
from pacer_core import train_predictor as train_mod
from pacer_core.sfm_generate_dataset import generate_pool


class CalibrationNode(Node):
    def __init__(self):
        super().__init__("pacer_calibration")

        self.declare_parameter("calibration_path", "")
        self.declare_parameter("n_mpc", 6)
        self.declare_parameter("alpha", 0.1)
        self.declare_parameter("delta", 0.1)
        self.declare_parameter("h_param", 12)
        self.declare_parameter("predictions_topic", "/pacer/pedestrian_predictions")
        self.declare_parameter("tracks_topic", "/pacer/pedestrian_tracks")
        self.declare_parameter("q_tupac_topic", "/pacer/q_tupac")

        self.n_mpc = int(self.get_parameter("n_mpc").value)
        self.alpha = float(self.get_parameter("alpha").value)
        self.delta = float(self.get_parameter("delta").value)
        self.h_param = int(self.get_parameter("h_param").value)
        self.alpha_win = self.alpha / self.h_param

        self.worker = None
        self._latest_pos = {}     # id -> (x,y), dall'ultimo PedestrianTrackArray
        self._mature_at = defaultdict(list)
        self._t_now = 0
        self._n_scored_total = 0
        self._n_censored_total = 0

        cal_path = self.get_parameter("calibration_path").value
        if cal_path and os.path.exists(cal_path):
            self._load_and_start_worker(cal_path)
        else:
            self.get_logger().warn(
                "nessuna calibrazione caricata (parametro 'calibration_path' vuoto o "
                "file assente) — q_tupac(k) NON disponibile finche' non si richiama "
                "il servizio RecalibrateTupac (§Sim2Real: calibrazione sempre "
                "disponibile su richiesta, mai implicita).")

        cb_online = ReentrantCallbackGroup()
        cb_services = ReentrantCallbackGroup()

        self.create_subscription(PedestrianPredictionArray,
                                    self.get_parameter("predictions_topic").value,
                                    self._on_predictions, 10, callback_group=cb_online)
        self.create_subscription(PedestrianTrackArray,
                                    self.get_parameter("tracks_topic").value,
                                    self._on_tracks, 10, callback_group=cb_online)
        self.q_pub = self.create_publisher(QTupacSnapshot,
                                             self.get_parameter("q_tupac_topic").value, 10)

        self.create_service(TrainPredictor, "/pacer/train_predictor", self._on_train,
                              callback_group=cb_services)
        self.create_service(RecalibrateTupac, "/pacer/recalibrate_tupac", self._on_recalibrate,
                              callback_group=cb_services)

        self.get_logger().info("pacer_calibration avviato.")

    # ---------- calibrazione online ----------

    def _on_tracks(self, msg):
        for t in msg.tracks:
            if t.history:
                p = t.history[-1]
                self._latest_pos[t.id] = (p.x, p.y)

    def _on_predictions(self, msg):
        if self.worker is None:
            return   # nessuna calibrazione caricata: niente da aggiornare

        # 1) risolvi le maturazioni schedulate per l'attuale t_now
        events = self._mature_at.pop(self._t_now, [])
        for k, pid, mu_k, Sigma_k in events:
            real_xy = self._latest_pos.get(pid)
            if real_xy is None:
                self._n_censored_total += 1
                continue
            r = mahalanobis_score(np.array(real_xy), mu_k, Sigma_k)
            self.worker.submit(k, r)
            self._n_scored_total += 1

        # 2) schedula i nuovi eventi da queste predizioni (k=1..n_mpc nel futuro)
        for pred in msg.predictions:
            if pred.n_steps < self.n_mpc:
                continue
            mu = np.array([[p.x, p.y] for p in pred.mu[:self.n_mpc]])
            Sigma = np.array(pred.sigma_flat[:self.n_mpc * 4]).reshape(self.n_mpc, 2, 2)
            for k in range(1, self.n_mpc + 1):
                self._mature_at[self._t_now + k].append((k, pred.id, mu[k - 1], Sigma[k - 1]))

        self._t_now += 1

        # 3) pubblica l'ultimo snapshot disponibile (mai un'attesa sul worker)
        q = self.worker.snapshot_q(self.n_mpc)
        out = QTupacSnapshot()
        out.header.stamp = self.get_clock().now().to_msg()
        out.k = list(range(1, self.n_mpc + 1))
        out.q = [q[k] for k in out.k]
        out.n_processed = self.worker.n_processed
        out.max_backlog = self.worker.max_backlog
        st = self.worker.staleness_s
        out.staleness_mean_ms = float(np.mean(st) * 1000) if st else 0.0
        self.q_pub.publish(out)

    def _load_and_start_worker(self, path):
        n0, pools, q_warmstart = calib_off.load_calibration(path)
        pool_sorted = {k: list(v) for k, v in pools.items()}
        s = {k: n0 for k in pools}
        h = HFunction("heavy_tail")
        if self.worker is not None:
            self.worker.stop_and_join()
        self.worker = OnlineCalibrationWorker(pool_sorted, s, q_warmstart, self.alpha_win,
                                                 self.delta, h)
        self.worker.start()
        self.get_logger().info(f"calibrazione caricata da {path} (n0={n0}), worker avviato.")

    # ---------- servizi on-demand ----------

    def _on_train(self, request, response):
        t0 = time.time()
        try:
            sfm_cfg = _default_sfm_cfg(request.scenario)
            windows = generate_pool(sfm_cfg, request.n_runs_train, request.n_peds_scene,
                                      seed0=request.seed * 1000 + 1, scenario=request.scenario)
            cfg = {"predictor": {"epochs": request.epochs, "lr": request.lr,
                                    "hidden_dim": request.hidden_dim}, "seed": request.seed}
            log_path = os.path.splitext(request.out_model_path)[0] + "_train_log.json"
            train_mod.train(windows, cfg, out_path=request.out_model_path, log_path=log_path,
                              seed=request.seed)
            import json
            final_nll = json.load(open(log_path))[-1]["nll"]
            response.success = True
            response.message = f"modello salvato in {request.out_model_path}"
            response.final_nll = float(final_nll)
        except Exception as e:   # noqa: BLE001 — servizio deve sempre rispondere, mai crashare il nodo
            response.success = False
            response.message = f"training fallito: {e}"
            response.final_nll = float("nan")
        response.elapsed_s = time.time() - t0
        return response

    def _on_recalibrate(self, request, response):
        t0 = time.time()
        try:
            model = SocialSTGCNN(obs_len=8, pred_len=request.h_param, hidden_dim=32)
            model.load_state_dict(torch.load(request.model_path, map_location="cpu"))
            model.eval()
            sfm_cfg = _default_sfm_cfg(request.scenario)
            n0, pools = calib_off.build_dcal_from_sfm(
                model, sfm_cfg, request.n_runs_cal, request.n_peds_scene,
                seed0=request.seed * 1000 + 1, H=request.h_param, scenario=request.scenario)
            alpha_win = request.alpha / request.h_param
            q_warmstart, _ = calib_off.quantile_warmstart(pools, n0, alpha_win, request.delta)
            calib_off.save_calibration(request.out_calibration_path, pools, n0, q_warmstart)
            self._load_and_start_worker(request.out_calibration_path + ".npz"
                                          if not request.out_calibration_path.endswith(".npz")
                                          else request.out_calibration_path)
            response.success = True
            response.message = f"calibrazione salvata in {request.out_calibration_path}"
            response.n0 = n0
            response.k = sorted(q_warmstart)
            response.q_warmstart = [q_warmstart[k] for k in response.k]
        except Exception as e:   # noqa: BLE001
            response.success = False
            response.message = f"ricalibrazione fallita: {e}"
            response.n0 = 0
            response.k = []
            response.q_warmstart = []
        response.elapsed_s = time.time() - t0
        return response


def _default_sfm_cfg(scenario):
    """Parametri SFM di default (stessi valori di config.yaml nella
    simulazione, §Sim2Real: D_cal ancora generato da simulazione, non da
    log Qualisys reali — vedi pacer_core.calibration_offline per il punto
    di estensione futuro)."""
    cfg = {
        "tau": 0.5, "A": 2.1, "B": 0.3, "A_wall": 5.0, "B_wall": 0.2, "radius": 0.3,
        "v0_mean": 1.34, "v0_std": 0.26, "dt_sim": 0.05, "dt_out": 0.4, "duration": 16.0,
        "corridor_length": 20.0, "corridor_width": 6.0,
    }
    if scenario == "open_field":
        cfg.update({"field_size": 18.0, "duration": 20.0})
    elif scenario == "cross":
        cfg.update({"field_size": 16.0})
    return cfg


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        if node.worker is not None:
            node.worker.stop_and_join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
