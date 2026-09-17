"""Servizio ComputeMpcCommand: un ciclo di NMPC (vincolo rigido o costo
soft), chiamato dal plugin Nav2 pacer_nav2_controller ad ogni ciclo del
controller server. Nodo VOLUTAMENTE senza stato tra una chiamata e l'altra
(il warm start u_warm e' responsabilita' del chiamante, cosi' come
raccogliere pedoni/q_tupac/ostacoli statici dai rispettivi topic — questo
nodo importa solo pacer_core.unicycle_nmpc / unicycle_nmpc_soft, stesso
solver gia' validato in simulazione, nessuna riscrittura della matematica).
"""
import numpy as np
import rclpy
from rclpy.node import Node

from pacer_msgs.srv import ComputeMpcCommand

from pacer_core import unicycle_nmpc as hard_nmpc
from pacer_core import unicycle_nmpc_soft as soft_nmpc
from pacer_core.obstacle_builder import build_obstacles_hard, build_obstacles_soft


class MpcSolverNode(Node):
    def __init__(self):
        super().__init__("pacer_mpc_solver")

        self.declare_parameter("dt", 0.4)
        self.declare_parameter("k_neigh", 6)
        self.declare_parameter("cutoff_m", 7.0)
        self.declare_parameter("weights", [6.0, 0.15, 0.02, 0.05])
        self.declare_parameter("lambda_soft", 1500.0)
        self.declare_parameter("beta", 1.0)
        self.declare_parameter("gamma_exp", 1.3)
        self.declare_parameter("a_gauss", 1.0)
        self.declare_parameter("d_safe_pedestrian", 2.0)
        self.declare_parameter("w_clear_pedestrian", 6000.0)
        self.declare_parameter("w_static", 4000.0)

        dt = float(self.get_parameter("dt").value)
        hard_nmpc.set_dt(dt)
        soft_nmpc.set_dt(dt)

        self.k_neigh = int(self.get_parameter("k_neigh").value)
        self.cutoff_m = float(self.get_parameter("cutoff_m").value)
        self.weights = tuple(self.get_parameter("weights").value)
        self.lambda_soft = float(self.get_parameter("lambda_soft").value)
        self.beta = float(self.get_parameter("beta").value)
        self.gamma_exp = float(self.get_parameter("gamma_exp").value)
        self.a_gauss = float(self.get_parameter("a_gauss").value)
        self.d_safe_ped = float(self.get_parameter("d_safe_pedestrian").value)
        self.w_clear_ped = float(self.get_parameter("w_clear_pedestrian").value)
        self.w_static = float(self.get_parameter("w_static").value)

        self.create_service(ComputeMpcCommand, "/pacer/compute_mpc_command", self._on_request)
        self.get_logger().info(f"pacer_mpc_solver avviato (dt={dt}s).")

    def _on_request(self, request, response):
        n_mpc = request.n_mpc
        x0 = np.array([request.robot_x, request.robot_y, request.robot_theta])
        goal = np.array([request.goal_x, request.goal_y])
        u_warm = (np.array(request.u_warm_flat).reshape(n_mpc, 2)
                   if len(request.u_warm_flat) == 2 * n_mpc else None)

        preds, positions_now = {}, {}
        for p in request.pedestrians.predictions:
            if p.n_steps < n_mpc:
                continue
            mu_abs = np.array([[pt.x, pt.y] for pt in p.mu[:n_mpc]])
            Sigma = np.array(p.sigma_flat[:n_mpc * 4]).reshape(n_mpc, 2, 2)
            preds[p.id] = (mu_abs, Sigma)
            positions_now[p.id] = np.array([p.current_xy.x, p.current_xy.y])

        static_obstacles = [{"xy": (o.xy.x, o.xy.y), "d_safe": o.d_safe}
                              for o in request.static_obstacles]

        robot_xy = x0[:2]
        q_by_k_dict = dict(zip(request.q_tupac.k, request.q_tupac.q))

        if request.controller_mode == "hard":
            # q_tupac(k) mancante per qualche k (nessuna calibrazione ancora
            # disponibile, §Sim2Real) -> vincolo vacuo per quel k (q2=inf),
            # mai un errore silenzioso: il ciclo restera' semplicemente
            # feasible su quell'ostacolo finche' la calibrazione non matura.
            q2_by_k = np.array([q_by_k_dict.get(k, np.inf) ** 2 for k in range(1, n_mpc + 1)])
            obstacles = build_obstacles_hard(preds, positions_now, robot_xy, q2_by_k,
                                                self.k_neigh, self.cutoff_m, n_mpc)
            u0, ok, u_full, margin, status = hard_nmpc.solve_step(
                x0, goal, obstacles, n_mpc, request.v_max, request.omega_max, u_warm=u_warm,
                weights=self.weights, static_obstacles=static_obstacles, w_static=self.w_static)
            response.success = True
            response.status = status
            response.v, response.omega = float(u0[0]), float(u0[1])
            response.u_full_flat = u_full.reshape(-1).tolist()
            response.min_margin = float(margin)
            response.cost_ped = float("nan")

        elif request.controller_mode == "soft":
            q_by_k = np.array([q_by_k_dict.get(k, np.inf) for k in range(1, n_mpc + 1)])
            obstacles = build_obstacles_soft(preds, positions_now, robot_xy, q_by_k,
                                                self.beta, self.gamma_exp, self.a_gauss,
                                                self.k_neigh, self.cutoff_m, n_mpc)
            u0, u_full, cost_ped, status = soft_nmpc.solve_step_soft(
                x0, goal, obstacles, n_mpc, request.v_max, request.omega_max, self.lambda_soft,
                u_warm=u_warm, weights=self.weights, d_safe=self.d_safe_ped,
                w_clear=self.w_clear_ped, static_obstacles=static_obstacles, w_static=self.w_static)
            response.success = True
            response.status = status
            response.v, response.omega = float(u0[0]), float(u0[1])
            response.u_full_flat = u_full.reshape(-1).tolist()
            response.min_margin = float("nan")
            response.cost_ped = float(cost_ped)

        else:
            response.success = False
            response.status = f"controller_mode sconosciuto: '{request.controller_mode}'"
            response.v = response.omega = 0.0
            response.u_full_flat = []
            response.min_margin = response.cost_ped = float("nan")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = MpcSolverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
