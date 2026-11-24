"""Test offline (senza ROS2) del porting Mondrian per distanza (rev. 2026-11-13, nota §3o).

rclpy, pacer_msgs e geometry_msgs sono sostituiti da stub minimi, cosi' i nodi VERI
(pacer_predictor._set_group, pacer_calibration._on_predictions/_on_tracks,
pacer_mpc_solver._solve) girano su Windows o in CI senza colcon build.

  1. calibrazione: stream sintetico in cui il predittore sbaglia di piu' vicino al
     robot (errore x3 a meno di d_near). Con mondrian_group="none" il gruppo vicino
     resta sotto-coperto; con "dist" ciascun gruppo torna vicino al nominale.
  2. MPC: lo stesso pedone riceve un vincolo piu' largo se e' nel gruppo vicino;
     gruppo ignoto -> quantile massimo.
  3. D_cal dalle tracce: dcal_from_tracks --d-near su un CSV sintetico produce i
     pool per gruppo e il nodo li carica.

Uso: python tools/test_mondrian_offline.py   (dalla radice di pacer-tupac-code-ros)
"""
import csv
import math
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "pacer")
for pkg in ("pacer_core", "pacer_calibration", "pacer_mpc_solver", "pacer_predictor"):
    sys.path.insert(0, os.path.join(SRC, pkg))


# ---------------------------------------------------------------- stub ROS2
class _Msg:
    _defaults = {}

    def __init__(self, **kw):
        for k, v in self._defaults.items():
            setattr(self, k, v() if callable(v) else v)
        for k, v in kw.items():
            setattr(self, k, v)


def _msg(name, **defaults):
    return type(name, (_Msg,), {"_defaults": defaults})


class _Stamp:
    def to_msg(self):
        return None


Header = _msg("Header", stamp=None, frame_id="")
Point = _msg("Point", x=0.0, y=0.0, z=0.0)
PointStamped = _msg("PointStamped", header=Header, point=Point)
PedestrianPrediction = _msg("PedestrianPrediction", id="", kind=0, current_xy=Point, n_steps=0,
                            mu=list, sigma_flat=list, radius=list, group=0, d_robot=float("nan"))
PedestrianPredictionArray = _msg("PedestrianPredictionArray", header=Header, predictions=list)
PedestrianTrack = _msg("PedestrianTrack", id="", history=list)
PedestrianTrackArray = _msg("PedestrianTrackArray", header=Header, tracks=list)
QTupacSnapshot = _msg("QTupacSnapshot", header=Header, k=list, q=list, n_groups=0, d_near=0.0,
                      n_processed=0, max_backlog=0, staleness_mean_ms=0.0)
StaticObstacle = _msg("StaticObstacle", xy=Point, d_safe=0.0)


class _Clock:
    t = 0.0

    def now(self):
        c = self

        class _N:
            nanoseconds = int(c.t * 1e9)

            def to_msg(self):
                return None
        return _N()


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, m):
        pass

    def warn(self, m):
        self.warnings.append(m)

    warning = warn

    def error(self, m):
        raise RuntimeError(m)


class _Param:
    def __init__(self, v):
        self.value = v


class Node:
    PARAMS = {}          # override per test

    def __init__(self, name):
        self._p = {}
        self._clock = _Clock()
        self._logger = _Logger()
        self.published = []

    def declare_parameter(self, n, default):
        self._p[n] = self.PARAMS.get(n, default)

    def get_parameter(self, n):
        return _Param(self._p[n])

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return self._clock

    def create_subscription(self, *a, **kw):
        return None

    def create_service(self, *a, **kw):
        return None

    def create_timer(self, *a, **kw):
        return None

    def create_publisher(self, *a, **kw):
        node = self

        class _P:
            def publish(self, m):
                node.published.append(m)
        return _P()


def _install_stubs():
    rclpy = types.ModuleType("rclpy")
    rclpy.node = types.ModuleType("rclpy.node")
    rclpy.node.Node = Node
    rclpy.callback_groups = types.ModuleType("rclpy.callback_groups")
    rclpy.callback_groups.ReentrantCallbackGroup = lambda: None
    rclpy.executors = types.ModuleType("rclpy.executors")
    rclpy.executors.MultiThreadedExecutor = object
    rclpy.qos = types.ModuleType("rclpy.qos")
    msgs = types.ModuleType("pacer_msgs")
    msgs.msg = types.ModuleType("pacer_msgs.msg")
    for c in (PedestrianPrediction, PedestrianPredictionArray, PedestrianTrack, PedestrianTrackArray,
              QTupacSnapshot, StaticObstacle):
        setattr(msgs.msg, c.__name__, c)
    msgs.srv = types.ModuleType("pacer_msgs.srv")
    for n in ("TrainPredictor", "RecalibrateTupac", "ComputeMpcCommand"):
        setattr(msgs.srv, n, object)
    geo = types.ModuleType("geometry_msgs")
    geo.msg = types.ModuleType("geometry_msgs.msg")
    geo.msg.Point, geo.msg.PointStamped = Point, PointStamped
    for name, mod in (("rclpy", rclpy), ("rclpy.node", rclpy.node),
                      ("rclpy.callback_groups", rclpy.callback_groups),
                      ("rclpy.executors", rclpy.executors), ("rclpy.qos", rclpy.qos),
                      ("pacer_msgs", msgs), ("pacer_msgs.msg", msgs.msg), ("pacer_msgs.srv", msgs.srv),
                      ("geometry_msgs", geo), ("geometry_msgs.msg", geo.msg)):
        sys.modules[name] = mod


_install_stubs()
from pacer_core import calibration_offline as calib_off      # noqa: E402
from pacer_core.mondrian import GROUP_UNKNOWN, assign_group  # noqa: E402

N_MPC = 6
D_NEAR = 2.5
NEAR_INFLATION = 3.0     # il predittore sbaglia 3 volte di piu' vicino al robot


def _fake_dcal(path, rng, grouped, n=4000):
    """D_cal sintetico: score = norma di una gaussiana standard 2D (Rayleigh), x3 per
    le finestre vicine al robot (20% del totale). grouped=True salva anche i pool per
    gruppo, come dcal_from_tracks --d-near su tracce raccolte con il robot in movimento."""
    near = rng.random(n) < 0.2
    R = {k: np.linalg.norm(rng.standard_normal((n, 2)), axis=1) * np.where(near, NEAR_INFLATION, 1.0)
         for k in range(1, N_MPC + 1)}
    pools = {k: sorted(v.tolist()) for k, v in R.items()}
    gp = {(g, k): R[k][near == bool(g)] for g in (0, 1) for k in R} if grouped else None
    calib_off.save_calibration(path, pools, n, {k: 3.0 for k in pools}, group_pools=gp)
    return path


def _miss_from_log(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for k in (1, 6):
        for near in (0, 1):
            sel = [r for r in rows if int(r["k"]) == k and int(int(float(r["d_robot"]) < D_NEAR) == near)
                   and float(r["q_used"]) > 0]
            m = np.mean([float(r["score"]) > float(r["q_used"]) for r in sel]) if sel else float("nan")
            out[(k, near)] = (100 * m, len(sel))
    return out


def test_calibration():
    """Stesso stream per tre configurazioni: un solo tracker; Mondrian con D_cal per
    gruppo; Mondrian con D_cal senza gruppi (warm start marginale, solo adattamento online)."""
    from pacer_calibration import calibration_node as cn
    tmp = tempfile.mkdtemp()
    dcal_g = _fake_dcal(os.path.join(tmp, "dcal_g.npz"), np.random.default_rng(0), grouped=True)
    dcal_m = _fake_dcal(os.path.join(tmp, "dcal_m.npz"), np.random.default_rng(0), grouped=False)
    cases = [("marginale", "none", dcal_g), ("Mondrian, D_cal per gruppo", "dist", dcal_g),
             ("Mondrian, D_cal senza gruppi", "dist", dcal_m)]
    res = {}
    for name, mode, dcal in cases:
        log = os.path.join(tmp, f"scores_{len(res)}.csv")
        Node.PARAMS = dict(calibration_path=dcal, calib_method="bet", mondrian_group=mode,
                           d_near=D_NEAR, score_log_csv=log, n_mpc=N_MPC)
        node = cn.CalibrationNode()
        _stream_static(node, mode, np.random.default_rng(1))
        node._log_f.close()
        snap = node.published[-1]
        res[name] = (_miss_from_log(log), np.array(snap.q).reshape(snap.n_groups, -1))
    print("\n[1] calibrazione BET, miss % (nominale 1.67 a k=1, 5 a k=6), vicini / lontani:")
    for name, (r, q) in res.items():
        print(f"  {name:30s} k=1 {r[(1, 1)][0]:5.2f} / {r[(1, 0)][0]:5.2f}   k=6 {r[(6, 1)][0]:5.2f} / "
              f"{r[(6, 0)][0]:5.2f}   q k=1 per gruppo {np.array2string(q[:, 0], precision=2)}")
    m, g = res["marginale"][0], res["Mondrian, D_cal per gruppo"][0]
    assert res["marginale"][1].shape[0] == 1 and res["Mondrian, D_cal per gruppo"][1].shape[0] == 2
    assert m[(6, 1)][0] > 10, "senza Mondrian il gruppo vicino dovrebbe essere sotto-coperto"
    assert g[(1, 1)][0] < 3.5 and g[(6, 1)][0] < 8, "con D_cal per gruppo il gruppo vicino va coperto"
    assert g[(1, 0)][0] < 1.67 + 0.5, "il gruppo lontano resta vicino al nominale"
    q_m, q_g = res["marginale"][1][0, 0], res["Mondrian, D_cal per gruppo"][1][0, 0]
    assert q_g < q_m / 1.5, "e i lontani ricevono regioni piu' piccole del caso marginale"
    print("  OK")


def _stream_static(node, mode, rng, T=1500, n_peds=10):
    """T cicli, pedoni fermi su un campo 12x12 m, robot che gira attorno al centro (i
    gruppi cambiano nel tempo). Previsione mu_k = posizione - errore, Sigma = I, quindi
    lo score a t+k e' ||errore_k||, con errore x3 se il pedone e' a meno di d_near."""
    pos = rng.uniform(0, 12, (n_peds, 2))
    for t in range(T):
        robot = np.array([6 + 4 * math.cos(t / 40), 6 + 4 * math.sin(t / 40)])
        trk = PedestrianTrackArray(tracks=[PedestrianTrack(id=f"p{i}", history=[Point(x=pos[i][0], y=pos[i][1])])
                                           for i in range(n_peds)])
        node._on_tracks(trk)
        arr = PedestrianPredictionArray(predictions=[])
        for i in range(n_peds):
            g, d = assign_group(mode, pos[i], robot, D_NEAR)
            scale = NEAR_INFLATION if d < D_NEAR else 1.0
            mu = pos[i][None] - rng.standard_normal((N_MPC, 2)) * scale
            arr.predictions.append(PedestrianPrediction(
                id=f"p{i}", kind=0, n_steps=N_MPC, current_xy=Point(x=pos[i][0], y=pos[i][1]),
                mu=[Point(x=a, y=b) for a, b in mu],
                sigma_flat=np.tile(np.eye(2).reshape(-1), N_MPC).tolist(), group=g, d_robot=d))
        node._on_predictions(arr)


def test_mpc():
    from pacer_mpc_solver import mpc_solver_node as mn
    import pacer_core.obstacle_builder as ob
    captured = {}
    orig = ob.build_obstacles_hard

    def spy(*a, **kw):
        out = orig(*a, **kw)
        captured["obs"] = out
        return out
    mn.build_obstacles_hard = spy
    Node.PARAMS = {}
    node = mn.MpcSolverNode()

    def pred(pid, x, g):
        return PedestrianPrediction(id=pid, kind=0, n_steps=N_MPC, current_xy=Point(x=x, y=3.0),
                                    mu=[Point(x=x, y=3.0)] * N_MPC,
                                    sigma_flat=np.tile((0.04 * np.eye(2)).reshape(-1), N_MPC).tolist(),
                                    group=g, d_robot=0.0)
    q0, q1 = [2.0] * N_MPC, [3.5] * N_MPC
    req = types.SimpleNamespace(
        controller_mode="hard", robot_x=0.0, robot_y=0.0, robot_theta=0.0, goal_x=10.0, goal_y=0.0,
        n_mpc=N_MPC, v_max=1.6, omega_max=2.0, u_warm_flat=[], static_obstacles=[],
        pedestrians=PedestrianPredictionArray(predictions=[pred("far", 4.0, 0), pred("near", 2.0, 1),
                                                           pred("unk", 6.0, GROUP_UNKNOWN)]),
        q_tupac=QTupacSnapshot(k=list(range(1, N_MPC + 1)), q=q0 + q1, n_groups=2, d_near=D_NEAR))
    resp = node._solve(req, types.SimpleNamespace())
    c = {o["id"]: math.sqrt(o["q2"][0]) - 0.6 / 0.2 for o in captured["obs"]}
    print(f"\n[2] MPC: q effettivo a k=1 per pedone: {', '.join(f'{k} {v:.2f}' for k, v in c.items())}"
          f"   (stato solver: {resp.status})")
    assert abs(c["far"] - 2.0) < 1e-3 and abs(c["near"] - 3.5) < 1e-3 and abs(c["unk"] - 3.5) < 1e-3
    # retro-compatibilita': snapshot a un gruppo
    req.q_tupac = QTupacSnapshot(k=list(range(1, N_MPC + 1)), q=q0, n_groups=1)
    node._solve(req, types.SimpleNamespace())
    c = {o["id"]: math.sqrt(o["q2"][0]) - 0.6 / 0.2 for o in captured["obs"]}
    assert all(abs(v - 2.0) < 1e-3 for v in c.values())
    print("  OK (anche con snapshot a un solo gruppo)")


def test_dcal_from_tracks():
    from pacer_core import dcal_from_tracks as dt
    from pacer_calibration import calibration_node as cn
    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "tracks.csv")
    rng = np.random.default_rng(3)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "stamp", "id", "x", "y", "fresh"])
        P = rng.uniform(0, 10, (4, 2))
        V = rng.normal(0, 0.4, (4, 2))
        for tick in range(400):
            rob = (5 + 3 * math.cos(tick / 30), 5 + 3 * math.sin(tick / 30))
            w.writerow([tick, f"{tick * 0.4:.1f}", "omnicar", f"{rob[0]:.3f}", f"{rob[1]:.3f}", 1])
            P += V * 0.4 + rng.normal(0, 0.03, P.shape)
            V[(P < 0) | (P > 10)] *= -1
            for i in range(4):
                w.writerow([tick, f"{tick * 0.4:.1f}", f"ped{i}", f"{P[i, 0]:.3f}", f"{P[i, 1]:.3f}", 1])
    model = os.path.join(HERE, "..", "..", "pacer-tupac-code", "13_closed_loop_benchmark", "results",
                         "predictor_open_field.pt")
    if not os.path.exists(model):
        print("\n[3] D_cal da tracce: predittore non trovato, salto")
        return
    n0, pools, q_ws, gp, G = dt.build_dcal_grouped(csv_path, model, N_MPC, "omnicar", D_NEAR)
    out = os.path.join(tmp, "dcal_g.npz")
    calib_off.save_calibration(out, pools, n0, q_ws, group_pools=gp)
    print(f"\n[3] D_cal da tracce: {n0} finestre, vicine {int((G == 1).sum())}, lontane {int((G == 0).sum())}")
    assert len(gp[(0, 1)]) + len(gp[(1, 1)]) == n0
    Node.PARAMS = dict(calibration_path=out, calib_method="bet", mondrian_group="dist", n_mpc=N_MPC)
    node = cn.CalibrationNode()
    assert set(node.trackers) == {(g, k) for g in (0, 1) for k in range(1, N_MPC + 1)}
    assert not [m for m in node.get_logger().warnings if "pool marginale" in m]
    print("  OK (pool per gruppo salvati e caricati dal nodo)")


if __name__ == "__main__":
    test_mpc()
    test_dcal_from_tracks()
    test_calibration()
    print("\ntutti i test passati")
