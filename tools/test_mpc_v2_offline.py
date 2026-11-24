"""Test offline (senza ROS2) di MPC v2 e filtro di sicurezza (rev. 2026-11-17, nota PACER §3r).

Usa gli stub di test_mondrian_offline.py (rclpy, pacer_msgs, geometry_msgs) piu' stub per
Twist e tf2_ros, e fa girare i nodi VERI:
  1. pacer_mpc_solver._solve in controller_mode "hard_v2": limiti di accelerazione dalla
     velocita' misurata, arresto terminale, vincoli rispettati anche fra i campioni.
  2. pacer_core.safety_layer.unsafe_people: persona che viene incontro -> non sicuro;
     persona che cammina parallela o robot fermo -> sicuro.
  3. pacer_mpc_solver.safety_filter_node: comando lasciato passare, frenata con persona sul
     percorso, frenata fail-safe con tracce vecchie o senza TF.

Uso: python tools/test_mpc_v2_offline.py   (dalla radice di pacer-tupac-code-ros; serve casadi)
"""
import math
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_mondrian_offline as T      # noqa: E402  (installa gli stub ROS2)

Twist = T._msg("Twist", linear=lambda: T.Point(), angular=lambda: T.Point())
geo = sys.modules["geometry_msgs.msg"]
geo.Twist = Twist
tf2 = types.ModuleType("tf2_ros")


class _TF:
    pose = (0.0, 0.0, 0.0)
    ok = True

    def __init__(self, *a, **kw):
        pass

    def lookup_transform(self, *a, **kw):
        if not _TF.ok:
            raise RuntimeError("no tf")
        x, y, th = _TF.pose
        tr = types.SimpleNamespace(translation=types.SimpleNamespace(x=x, y=y, z=0.0),
                                   rotation=types.SimpleNamespace(x=0.0, y=0.0, z=math.sin(th / 2),
                                                                  w=math.cos(th / 2)))
        return types.SimpleNamespace(transform=tr)


tf2.Buffer, tf2.TransformListener = _TF, (lambda *a, **kw: None)
sys.modules["tf2_ros"] = tf2
rclpy = sys.modules["rclpy"]
rclpy.time = types.ModuleType("rclpy.time")
rclpy.time.Time = lambda: None
sys.modules["rclpy.time"] = rclpy.time

N = 6
A_ACC, A_DEC, DT = 1.0, 1.5, 0.4


def test_hard_v2():
    from pacer_mpc_solver import mpc_solver_node as mn
    from pacer_core import unicycle_nmpc_v2 as v2
    T.Node.PARAMS = {}
    node = mn.MpcSolverNode()
    sig = np.tile((0.04 * np.eye(2)).reshape(-1), N).tolist()
    # persona che attraversa davanti al robot da destra a sinistra
    mu = [T.Point(x=2.5, y=-1.5 + 0.5 * (k + 1)) for k in range(N)]
    ped = T.PedestrianPrediction(id="p", kind=0, n_steps=N, current_xy=T.Point(x=2.5, y=-1.5), mu=mu,
                                 sigma_flat=sig, group=0, d_robot=2.9)
    req = types.SimpleNamespace(
        controller_mode="hard_v2", robot_x=0.0, robot_y=0.0, robot_theta=0.0, robot_v=1.0, robot_omega=0.0,
        goal_x=10.0, goal_y=0.0, n_mpc=N, v_max=1.6, omega_max=2.0, u_warm_flat=[], static_obstacles=[],
        pedestrians=T.PedestrianPredictionArray(predictions=[ped]),
        q_tupac=T.QTupacSnapshot(k=list(range(1, N + 1)), q=[2.0] * N, n_groups=1, d_near=2.5))
    t0 = time.perf_counter()
    resp = node._solve(req, types.SimpleNamespace())
    ms = 1e3 * (time.perf_counter() - t0)
    u = np.array(resp.u_full_flat).reshape(N, 2)
    print(f"[1] hard_v2: stato {resp.status}, {ms:.0f} ms, piano v = {np.round(u[:, 0], 2).tolist()}")
    assert resp.status in ("optimal", "fallback_stop", "fallback_evasive")
    dv = np.diff(np.concatenate([[1.0], u[:, 0]]))
    assert np.all(dv <= A_ACC * DT + 1e-6) and np.all(dv >= -A_DEC * DT - 1e-6), dv
    assert abs(u[-1, 0]) < 1e-6, "arresto terminale"
    # vincolo fra i campioni: distanza di Mahalanobis su 30 punti per intervallo, robot in moto
    c = 2.0 + 0.6 / 0.2
    px, py, th = 0.0, 0.0, 0.0
    prev = np.array([2.5, -1.5]); worst = np.inf
    for k in range(N):
        v, w = u[k]
        nxt = np.array([px + v * math.cos(th) * DT, py + v * math.sin(th) * DT])
        muk = np.array([mu[k].x, mu[k].y])
        for s in np.linspace(0, 1, 31)[1:]:
            r = (np.array([px, py]) + s * (nxt - [px, py])) - (prev + s * (muk - prev))
            if v > 0.05:
                worst = min(worst, float(np.sqrt(r @ r / 0.04)) - c)
        px, py, th = nxt[0], nxt[1], th + w * DT
        prev = muk
    print(f"    margine minimo fra i campioni (unita' di Mahalanobis): {worst:.3f}")
    assert worst > -0.05
    print("    OK (accelerazione, arresto terminale, vincoli fra i campioni)")
    v2.configure(SOCIAL=False)


def test_safety_layer():
    from pacer_core.safety_layer import unsafe_people
    x = np.array([0.0, 0.0, 0.0])
    head_on = unsafe_people(x, 1.0, 0.0, [[2.0, 0.0]], [[-1.0, 0.0]])
    parallel = unsafe_people(x, 1.0, 0.0, [[1.0, 1.5]], [[1.0, 0.0]])
    stopped = unsafe_people(x, 0.0, 0.0, [[0.7, 0.0]], [[-1.0, 0.0]])
    passing = unsafe_people(x, 1.0, 0.0, [[2.0, 1.2]], [[-1.0, 0.0]])
    print(f"[2] safety_layer: incontro {head_on}, parallela {parallel}, robot fermo {stopped}, "
          f"incrocio a 1.2 m di lato {passing}")
    assert head_on == [0] and parallel == [] and stopped == [] and passing == []
    # con deviazione possibile (a_ped) l'incrocio laterale diventa non sicuro
    assert unsafe_people(x, 1.0, 0.0, [[2.0, 1.2]], [[-1.0, 0.0]], a_ped=1.0) == [0]
    print("    OK")


def test_safety_filter():
    from pacer_mpc_solver import safety_filter_node as sf
    T.Node.PARAMS = {}
    node = sf.SafetyFilterNode()

    def tracks(pts):
        return T.PedestrianTrackArray(tracks=[T.PedestrianTrack(id=f"p{i}", history=[T.Point(x=a, y=b)
                                                                                      for a, b in h])
                                              for i, h in enumerate(pts)])

    def cmd(v, w=0.0):
        m = Twist(); m.linear.x, m.angular.z = v, w
        node._on_cmd(m)
        return node.published[-1].linear.x, node.published[-1].angular.z

    _TF.pose, _TF.ok = (0.0, 0.0, 0.0), True
    node._on_tracks(tracks([[(5.0, 3.0), (5.0, 3.0)]]))          # persona lontana, ferma
    out1 = cmd(1.0)
    node.v_last = 1.0
    node._on_tracks(tracks([[(2.4, 0.0), (2.0, 0.0)]]))          # viene incontro a 1 m/s
    node.t_last = time.monotonic() - 0.4
    out2 = cmd(1.2)
    node.t_tracks = time.monotonic() - 5.0                       # tracce vecchie
    node.v_last, node.t_last = 1.0, time.monotonic() - 0.4
    out3 = cmd(1.0)
    _TF.ok = False
    node._on_tracks(tracks([[(5.0, 3.0), (5.0, 3.0)]]))
    node.v_last, node.t_last = 1.0, time.monotonic() - 0.4
    out4 = cmd(1.0)
    print(f"[3] filtro: libero {out1}, persona incontro {np.round(out2, 2)}, tracce vecchie "
          f"{np.round(out3, 2)}, senza TF {np.round(out4, 2)}; eventi {node.n_events}")
    assert out1 == (1.0, 0.0)
    assert abs(out2[0] - 0.4) < 0.05 and out2[1] == 0.0       # 1.0 - 1.5*0.4
    assert out3[0] < 1.0 and out4[0] < 1.0
    print("    OK")


if __name__ == "__main__":
    test_safety_layer()
    test_safety_filter()
    test_hard_v2()
    print("\nTutti i test passati.")
