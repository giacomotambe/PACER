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
import csv
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

from pacer_core.calibration_trackers import make_tracker
from pacer_core.mondrian import GROUP_UNKNOWN, n_groups as mondrian_n_groups
from pacer_core.nonconformity import mahalanobis_score
from pacer_core.quantile_tupac import HFunction
from pacer_core import calibration_offline as calib_off
from pacer_core.social_stgcnn import SocialSTGCNN
from pacer_core import train_predictor as train_mod
from pacer_core.sfm_generate_dataset import generate_pool


# rev. 2026-11-24: calibratori predefiniti, scelti con calib_preset (config/calibrator.yaml) o dal launch
# (calib_preset:=...). Ogni preset imposta (calib_method a k=1, calib_method_long a k>=2, alpha_far).
# Risultati in closed loop: nota PACER §3s, run v48-v50 (stress = persone imprevedibili, alpha_1 = 1.67%).
CALIB_PRESETS = {
    # standard dal 2026-11-23 (v50 mxg_B): BET a miscela a k=1, quantile Mondrian a k>=2, alpha 0.1 ai lontani.
    # miss vicini k=1 sotto stress 1.48%; q vicini k=1 -6/8% rispetto a "conservative".
    "standard": ("xbet", "static", 0.1),
    # piu' conservativo (v50 mfbet_B): BET a capitale limitato, una sola puntata. Miss sotto stress 1.29%.
    "conservative": ("fbet", "static", 0.1),
    # standard precedente (v44 base): BET a ogni offset, nessun alpha separato per i lontani.
    "legacy": ("bet", "", -1.0),
    # baseline senza garanzie anytime, per confronto: quantile fisso di D_cal a ogni offset.
    "static": ("static", "", -1.0),
}


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
        # rev. 2026-11-04: condizione sperimentale e log degli score
        # rev. 2026-11-24: calib_preset = standard | conservative | legacy | static | custom.
        # Con un preset diverso da "custom" i tre parametri sotto (calib_method, calib_method_long, alpha_far)
        # sono ignorati. calib_preset_override (dal launch, "" = nessuno) ha la precedenza sul file.
        self.declare_parameter("calib_preset", "standard")
        self.declare_parameter("calib_preset_override", "")
        self.declare_parameter("calib_method", "xbet")     # none|static|aci|kl|bet|fbet|xbet (xbet dal 2026-11-23)
        # rev. 2026-11-23 (run v48/v50, nota §3s): calibratore per gli offset k>=2, che servono solo alla
        # pianificazione (il Teorema 3 usa k=1). "" = come calib_method; "static" = quantile Mondrian di D_cal
        # (standard: regioni k>=2 -20/30%, tempo -2/14 s, garanzia k=1 invariata); "aci" sconsigliato (esplode sotto stress).
        self.declare_parameter("calib_method_long", "static")
        self.declare_parameter("score_log_csv", "")        # vuoto = nessun log
        # rev. 2026-11-11: ripartizione di alpha/delta sugli offset (benchmark §3h, run v7).
        # 'front': il Teorema 3 usa solo k=1 -> alpha_1 = alpha/n_mpc e delta pieno a k=1;
        # offset k>=2 (vincoli di pianificazione) con alpha_long e delta pieno ciascuno.
        # 'uniform': alpha/n_mpc e delta/n_mpc per ogni offset (comportamento precedente).
        self.declare_parameter("alpha_split", "front")      # front | uniform
        self.declare_parameter("alpha_long", 0.1)   # 0.1 dal 2026-11-19 (nota PACER §3s)
        # rev. 2026-11-13: Mondrian per distanza dal robot (pacer_core/mondrian.py, nota §3o).
        # Deve coincidere con mondrian_group di pacer_predictor, che assegna il gruppo.
        # Un tracker per (gruppo, k), delta/G per gruppo.
        self.declare_parameter("mondrian_group", "none")   # none | dist
        self.declare_parameter("d_near", 2.5)
        # rev. 2026-11-22 (nota §3s, prove A e B):
        #  sigma_pow: Sigma^g usato quando il nodo ricostruisce D_cal (servizi), uguale a pacer_predictor;
        #  alpha_far: alpha a k=1 del gruppo 'lontani' (g=0, d >= d_near); < 0 = alpha_1 come i vicini.
        #  I lontani non raggiungono il robot in un ciclo (0.6 + (2.5+1.6)*0.4 = 2.24 m < d_near):
        #  la garanzia del Teorema 3 sui vicini non cambia, l'area k=1 cala del 30-45%.
        self.declare_parameter("sigma_pow", 1.0)
        self.declare_parameter("sigma_pow_from_k", 1)   # 2 = k=1 invariato (rev. 2026-11-22, v45)
        self.declare_parameter("alpha_far", -1.0)

        self.n_mpc = int(self.get_parameter("n_mpc").value)
        self.alpha = float(self.get_parameter("alpha").value)
        self.delta = float(self.get_parameter("delta").value)
        self.h_param = int(self.get_parameter("h_param").value)
        # rischio per offset sull'orizzonte effettivamente pianificato (n_mpc),
        # come nel benchmark closed-loop; h_param resta per i servizi offline
        self.alpha_win = self.alpha / self.n_mpc
        self.alpha_split = str(self.get_parameter("alpha_split").value)
        self.alpha_long = float(self.get_parameter("alpha_long").value)
        if self.alpha_split not in ("front", "uniform"):
            raise ValueError(f"alpha_split sconosciuto: '{self.alpha_split}' (front|uniform)")
        self.calib_preset = (str(self.get_parameter("calib_preset_override").value)
                             or str(self.get_parameter("calib_preset").value))
        if self.calib_preset == "custom":
            self.calib_method = str(self.get_parameter("calib_method").value)
            self.calib_method_long = str(self.get_parameter("calib_method_long").value) or self.calib_method
            self.alpha_far = float(self.get_parameter("alpha_far").value)
        elif self.calib_preset in CALIB_PRESETS:
            m1, mlong, af = CALIB_PRESETS[self.calib_preset]
            self.calib_method, self.calib_method_long, self.alpha_far = m1, (mlong or m1), af
        else:
            raise ValueError(f"calib_preset sconosciuto: '{self.calib_preset}' "
                             f"({' | '.join(list(CALIB_PRESETS) + ['custom'])})")
        self.mondrian_group = str(self.get_parameter("mondrian_group").value)
        self.n_groups = mondrian_n_groups(self.mondrian_group)
        self.d_near = float(self.get_parameter("d_near").value)
        from pacer_core.predictor_utils import set_sigma_pow
        set_sigma_pow(self.get_parameter("sigma_pow").value, self.get_parameter("sigma_pow_from_k").value)
        if self.mondrian_group == "none" and self.calib_preset in ("standard", "conservative"):
            self.get_logger().warn(f"calib_preset '{self.calib_preset}' e' validato con mondrian_group 'dist' "
                                   "(alpha_far non ha effetto senza gruppi).")
        self.get_logger().info(f"calibratore: preset '{self.calib_preset}' -> k=1 '{self.calib_method}', "
                               f"k>=2 '{self.calib_method_long}', alpha_far {self.alpha_far}")
        self._n_ungrouped_total = 0
        self.trackers = None
        self._rng = np.random.default_rng(0)
        self._log = None
        log_path = self.get_parameter("score_log_csv").value
        if log_path:
            self._log_f = open(log_path, "a", newline="")
            self._log = csv.writer(self._log_f)
            if self._log_f.tell() == 0:
                self._log.writerow(["t_issue", "k", "ped_id", "score", "q_used", "method", "stamp",
                                    "group", "d_robot"])

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
        if self.trackers is None:
            return   # nessuna calibrazione caricata: niente da aggiornare

        # 1) risolvi le maturazioni schedulate per l'attuale t_now, a batch per
        #    (offset, issue time): i pedoni simultanei entrano insieme (Sez. IV-B)
        events = self._mature_at.pop(self._t_now, [])
        batches = defaultdict(list)
        stamp = self.get_clock().now().nanoseconds * 1e-9
        for k, pid, mu_k, Sigma_k, t_iss, q_used, g, d_rob in events:
            real_xy = self._latest_pos.get(pid)
            if real_xy is None:
                self._n_censored_total += 1
                continue
            r = mahalanobis_score(np.array(real_xy), mu_k, Sigma_k)
            if self._log is not None:
                self._log.writerow([t_iss, k, pid, f"{r:.6f}", f"{q_used:.6f}",
                                    self.calib_method, f"{stamp:.3f}", g, f"{d_rob:.3f}"])
            if g == GROUP_UNKNOWN or g >= self.n_groups:
                self._n_ungrouped_total += 1     # gruppo ignoto: nessun tracker da aggiornare
                continue
            batches[(g, k, t_iss)].append(r)     # batch per (gruppo, offset, issue time)
            self._n_scored_total += 1
        for (g, k, t_iss), rs in sorted(batches.items(), key=lambda kv: kv[0][2]):
            self.trackers[(g, k)].update(t_iss, rs)
        if self._log is not None and events:
            self._log_f.flush()

        q_now = {(g, k): float(self.trackers[(g, k)].q())
                 for g in range(self.n_groups) for k in range(1, self.n_mpc + 1)}

        # 2) schedula i nuovi eventi da queste predizioni (k=1..n_mpc nel futuro)
        for pred in msg.predictions:
            if pred.kind != 0 or pred.n_steps < self.n_mpc:
                continue   # rev. 2026-11-11: solo le predizioni del modello entrano in TUPAC
            mu = np.array([[p.x, p.y] for p in pred.mu[:self.n_mpc]])
            Sigma = np.array(pred.sigma_flat[:self.n_mpc * 4]).reshape(self.n_mpc, 2, 2)
            g = int(pred.group) if self.n_groups > 1 else 0
            for k in range(1, self.n_mpc + 1):
                q_used = (q_now[(g, k)] if g < self.n_groups else
                          max(q_now[(gg, k)] for gg in range(self.n_groups)))
                self._mature_at[self._t_now + k].append(
                    (k, pred.id, mu[k - 1], Sigma[k - 1], self._t_now, q_used, g, float(pred.d_robot)))

        self._t_now += 1

        # 3) pubblica i quantili correnti (aggiornamento sincrono: pochi ms,
        #    il nodo e' separato dal solver MPC e non lo blocca)
        out = QTupacSnapshot()
        out.header.stamp = self.get_clock().now().to_msg()
        out.k = list(range(1, self.n_mpc + 1))
        # ordine gruppo-maggiore: q[g*n_mpc + k-1] (n_groups = 1: come prima)
        out.q = [float(self.trackers[(g, k)].q()) for g in range(self.n_groups) for k in out.k]
        out.n_groups = self.n_groups
        out.d_near = self.d_near
        out.n_processed = self._n_scored_total
        out.max_backlog = 0
        out.staleness_mean_ms = 0.0
        self.q_pub.publish(out)

    def _load_and_start_worker(self, path):
        n0, pools, _ = calib_off.load_calibration(path)
        gpools = calib_off.load_group_pools(path)        # {(g, k): pool}, {} se assenti
        trackers = {}
        for g in range(self.n_groups):
            for k in range(1, self.n_mpc + 1):
                marg = np.asarray(pools[k], dtype=float)
                pool = marg
                if self.n_groups > 1:
                    if len(gpools.get((g, k), [])) > 0:
                        pool = np.asarray(gpools[(g, k)], dtype=float)
                    elif k == 1:
                        self.get_logger().warn(
                            f"D_cal senza pool per il gruppo {g}: uso il pool marginale. Il "
                            "tracker si adatta online, ma il warm start non distingue i gruppi "
                            "(rigenera D_cal dalle tracce con robot: dcal_from_tracks --d-near).")
                # griglia BET: copre il pool marginale e quello di ogni gruppo
                top = max([float(np.quantile(marg, 0.999))] +
                          [float(np.quantile(v, 0.999)) for (gg, kk), v in gpools.items()
                           if kk == k and len(v)])
                grid = np.linspace(0.0, 2.0 * top, 800)
                a_k, d_k = self._alpha_delta(k)
                if k == 1 and g == 0 and self.n_groups > 1 and self.alpha_far > 0:
                    a_k = self.alpha_far          # lontani (rev. 2026-11-22)
                m_k = self.calib_method if k == 1 else self.calib_method_long
                trackers[(g, k)] = make_tracker(m_k, pool, k, self.n_mpc, a_k,
                                                d_k / self.n_groups, self._rng, grid=grid)
        self.trackers = trackers
        self.get_logger().info(
            f"calibrazione caricata da {path} (n0={n0}), metodo '{self.calib_method}' a k=1, "
            f"'{self.calib_method_long}' a k>=2, "
            f"split '{self.alpha_split}': alpha_1 = {self.alpha_win:.4f}, alpha_k>=2 = "
            f"{self._alpha_delta(2)[0]:.4f}, mondrian '{self.mondrian_group}' "
            f"({self.n_groups} gruppi, delta/{self.n_groups} per gruppo).")

    def _alpha_delta(self, k):
        """(alpha_k, delta passato a make_tracker). Interleaved divide delta per
        n_mpc*k: con 'front' si passa delta*n_mpc, cosi' ogni offset ha delta pieno
        (diviso solo tra le sue k fasi), come nel benchmark (PACER_SPLIT=front)."""
        if self.alpha_split == "uniform":
            return self.alpha_win, self.delta
        return (self.alpha_win if k == 1 else self.alpha_long), self.delta * self.n_mpc

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
        if node._log is not None:
            node._log_f.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
