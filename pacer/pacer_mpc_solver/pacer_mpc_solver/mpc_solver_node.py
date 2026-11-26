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
try:   # MPC v2 (rev. 2026-11-17, nota PACER §3r): richiede casadi (pip install casadi)
    from pacer_core import unicycle_nmpc_v2 as v2_nmpc
except ImportError:     # pragma: no cover
    v2_nmpc = None
from pacer_core.obstacle_builder import (build_obstacles_hard, build_obstacles_soft,
                                        build_unpredicted_hard, unpredicted_as_static)
from pacer_core.mondrian import unpack_q, q_for_group


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
        # rev. 2026-11-04: L = raggio robot + corpo pedone [m]; fallback evasivo
        self.declare_parameter("clearance_l", 0.6)
        self.declare_parameter("evasive", True)
        self.declare_parameter("status_log_csv", "")   # un record per ciclo MPC
        # MPC v2 (controller_mode "hard_v2"): limiti di accelerazione del robot reale, arresto
        # terminale, multi-start, costo sociale opzionale (nota PACER §3r)
        self.declare_parameter("v2_a_acc", 1.0)          # [m/s^2]
        self.declare_parameter("v2_a_dec", 1.5)          # [m/s^2] = decelerazione dello stop
        self.declare_parameter("v2_alpha", 3.0)          # [rad/s^2]
        self.declare_parameter("v2_terminal_stop", True)
        self.declare_parameter("v2_multistart", "lazy")  # "0" | "one" | "lazy" | "full"
        self.declare_parameter("v2_max_iter", 50)        # iterazioni IPOPT per avvio, oltre: ripiego
        self.declare_parameter("v2_t_budget", 0.25)      # [s] niente nuovi avvii oltre questo tempo
        # JIT (nota §3s): funzioni del problema compilate in C con gcc, cache su disco. Alla prima
        # accensione compila 11 librerie (minuti); poi le ricarica. Senza gcc: interpretato.
        self.declare_parameter("v2_jit", False)
        self.declare_parameter("v2_jit_flags", "-O1")
        self.declare_parameter("v2_jit_dir", "")         # "" = ~/.cache/pacer_mpc_jit
        self.declare_parameter("v2_jit_horizon", 6)      # N da precompilare (= n_mpc del controller)
        self.declare_parameter("v2_prune", True)         # potatura esatta dei vincoli irraggiungibili
        # rev. 2026-11-19 (nota §3s): vincoli a penalita' esatta e scadenza fissa
        self.declare_parameter("v2_soft", True)          # g >= -s, s >= 0, costo w_slack * s
        self.declare_parameter("v2_w_slack", 1000.0)
        self.declare_parameter("v2_deadline", 0.25)      # [s] oltre: piano precedente traslato o frenata; 0 = off
        # (< service_timeout_s del plugin, 0.35 s, meno la preparazione degli ostacoli)
        self.declare_parameter("v2_precompile", True)    # costruisce tutti i solutori all'avvio
        self.declare_parameter("v2_prev_fallback", True)  # senza piano: prima il piano precedente verificato
        self.declare_parameter("v2_social", False)
        self.declare_parameter("v2_w_comfort", 2.0)
        self.declare_parameter("v2_d_comfort", 1.2)
        self.declare_parameter("v2_w_side", 0.5)
        self.declare_parameter("v2_w_front", 1.0)
        self.declare_parameter("v2_w_path", 0.5)
        # rev. 2026-11-26 (benchmark v53-v60, prova NUC): costo di occupazione. Penalizza lo stare nella
        # regione predetta di una persona anche da fermo (il vincolo e' rilasciato a robot fermo): il
        # robot si scansa invece di fermarsi sulla linea di chi gli viene incontro. 0 = spento.
        self.declare_parameter("v2_w_occ", 0.0)
        self.declare_parameter("v2_occ_infl", 1.3)        # fattore sul raggio della regione
        self.declare_parameter("v2_occ_approach", True)   # conta solo chi si avvicina o e' fermo
        self.declare_parameter("v2_occ_goal_r", 0.0)      # [m] sfuma vicino al goal; 0 in ROS (goal = lookahead)
        # selettore aspettare/aggirare (sperimentale: sul NUC non migliora rispetto alla sola occupazione)
        self.declare_parameter("v2_strategy", False)

        dt = float(self.get_parameter("dt").value)
        hard_nmpc.set_dt(dt)
        soft_nmpc.set_dt(dt)
        if v2_nmpc is not None:
            gp = lambda n: self.get_parameter(n).value
            v2_nmpc.set_dt(dt)
            v2_nmpc.configure(
                A_ACC=float(gp("v2_a_acc")), A_DEC=float(gp("v2_a_dec")), ALPHA=float(gp("v2_alpha")),
                TERMINAL_STOP=bool(gp("v2_terminal_stop")), MULTISTART=str(gp("v2_multistart")),
                SOCIAL=bool(gp("v2_social")), W_COMFORT=float(gp("v2_w_comfort")),
                D_COMFORT=float(gp("v2_d_comfort")), W_SIDE=float(gp("v2_w_side")),
                W_FRONT=float(gp("v2_w_front")), W_PATH=float(gp("v2_w_path")),
                W_STATIC=float(gp("w_static")), PRUNE=bool(gp("v2_prune")),
                MAX_ITER=int(gp("v2_max_iter")), T_BUDGET=float(gp("v2_t_budget")),
                SOFT=bool(gp("v2_soft")), W_SLACK=float(gp("v2_w_slack")),
                PREV_FALLBACK=bool(gp("v2_prev_fallback")),
                WEIGHTS=tuple(float(w) for w in gp("weights")),
                W_OCC=float(gp("v2_w_occ")), OCC_INFL=float(gp("v2_occ_infl")),
                OCC_APPROACH=bool(gp("v2_occ_approach")), OCC_GOAL_R=float(gp("v2_occ_goal_r")),
                STRATEGY=bool(gp("v2_strategy")), STRAT_FAST=True, STRAT_LANES=True, STRAT_TAU=2.5,
                STRAT_R_FAR=1.2, STRAT_ACT=0.3, STRAT_TSUB=7.2)
            self.get_logger().info(f"MPC v2: costo di occupazione w_occ={float(gp('v2_w_occ'))}, "
                                   f"selettore={'on' if bool(gp('v2_strategy')) else 'off'}")
            if bool(gp("v2_jit")):
                jkw = {"JIT": True, "JIT_FLAGS": str(gp("v2_jit_flags"))}
                if str(gp("v2_jit_dir")):
                    jkw["JIT_DIR"] = str(gp("v2_jit_dir"))
                v2_nmpc.configure(**jkw)
                self.get_logger().info("MPC v2 JIT: precompilazione (minuti la prima volta)...")
            if bool(gp("v2_jit")) or bool(gp("v2_precompile")):
                # tutte le taglie (persone x statici, nessun semipiano): un solutore costruito durante
                # la navigazione costerebbe 0.1-2 s in un ciclo
                t_c = v2_nmpc.precompile(int(gp("v2_jit_horizon")), halfplanes=(0,))
                if v2_nmpc._JIT_FAILED:
                    self.get_logger().warn(f"JIT fallito, solutore interpretato: {v2_nmpc._JIT_FAILED[0][:300]}")
                self.get_logger().info(f"MPC v2: solutori pronti in {t_c:.0f} s")
            dl = float(gp("v2_deadline"))
            if dl <= 0 and bool(gp("v2_prev_fallback")):
                dl = 1e9                   # nessuna scadenza, ma il piano precedente resta la prima riserva
            self.v2_solve = v2_nmpc.DeadlineSolver(dl) if dl > 0 else v2_nmpc.solve_step
        else:
            self.get_logger().warn("casadi non installato: controller_mode 'hard_v2' non disponibile.")

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
        self.clearance_l = float(self.get_parameter("clearance_l").value)
        self.evasive = bool(self.get_parameter("evasive").value)
        self._log = None
        log_path = self.get_parameter("status_log_csv").value
        if log_path:
            import csv
            self._log_f = open(log_path, "a", newline="")
            self._log = csv.writer(self._log_f)
            if self._log_f.tell() == 0:
                self._log.writerow(["stamp", "status", "v", "omega", "min_margin",
                                    "robot_x", "robot_y", "n_peds", "solve_ms"])

        self.create_service(ComputeMpcCommand, "/pacer/compute_mpc_command", self._on_request)
        self.get_logger().info(f"pacer_mpc_solver avviato (dt={dt}s).")

    def _on_request(self, request, response):
        import time
        t0 = time.perf_counter()
        response = self._solve(request, response)
        if self._log is not None:
            self._log.writerow([f"{self.get_clock().now().nanoseconds * 1e-9:.3f}", response.status,
                                f"{response.v:.4f}", f"{response.omega:.4f}", f"{response.min_margin:.4f}",
                                f"{request.robot_x:.4f}", f"{request.robot_y:.4f}",
                                len(request.pedestrians.predictions),
                                f"{(time.perf_counter() - t0) * 1000:.1f}"])
            self._log_f.flush()
        return response

    def _solve(self, request, response):
        n_mpc = request.n_mpc
        x0 = np.array([request.robot_x, request.robot_y, request.robot_theta])
        goal = np.array([request.goal_x, request.goal_y])
        u_warm = (np.array(request.u_warm_flat).reshape(n_mpc, 2)
                   if len(request.u_warm_flat) == 2 * n_mpc else None)

        preds, positions_now, unpred, groups, unpred_now = {}, {}, {}, {}, {}
        for p in request.pedestrians.predictions:
            if p.n_steps < n_mpc:
                continue
            if p.kind != 0:      # rev. 2026-11-11: pedone con storico corto
                unpred[p.id] = (np.array([[pt.x, pt.y] for pt in p.mu[:n_mpc]]),
                                np.array(p.radius[:n_mpc], dtype=float))
                unpred_now[p.id] = np.array([p.current_xy.x, p.current_xy.y])
                continue
            mu_abs = np.array([[pt.x, pt.y] for pt in p.mu[:n_mpc]])
            Sigma = np.array(p.sigma_flat[:n_mpc * 4]).reshape(n_mpc, 2, 2)
            preds[p.id] = (mu_abs, Sigma)
            groups[p.id] = int(p.group)
            positions_now[p.id] = np.array([p.current_xy.x, p.current_xy.y])

        static_obstacles = [{"xy": (o.xy.x, o.xy.y), "d_safe": o.d_safe}
                              for o in request.static_obstacles]

        robot_xy = x0[:2]
        # rev. 2026-11-13 (Mondrian): quantili per gruppo, (G, n_mpc). q_tupac(k) mancante
        # (nessuna calibrazione ancora disponibile) -> q=inf -> vincolo SEMPRE violato su
        # quell'ostacolo -> fallback (stop/evasivo) o "infeasible": la direzione sicura.
        q_grp = unpack_q(request.q_tupac.k, request.q_tupac.q, request.q_tupac.n_groups, n_mpc)
        q_by_id = {i: q_for_group(q_grp, g) for i, g in groups.items()}
        q_by_k = q_grp.max(axis=0)       # non usato se q_by_id copre tutti i pedoni

        if request.controller_mode == "hard":
            obstacles = build_obstacles_hard(preds, positions_now, robot_xy, q_by_k,
                                                self.k_neigh, self.cutoff_m, n_mpc,
                                                clearance_l=self.clearance_l, q_by_id=q_by_id)
            obstacles += build_unpredicted_hard(unpred, robot_xy, self.cutoff_m, n_mpc,
                                                clearance_l=self.clearance_l)
            u0, ok, u_full, margin, status = hard_nmpc.solve_step(
                x0, goal, obstacles, n_mpc, request.v_max, request.omega_max, u_warm=u_warm,
                weights=self.weights, static_obstacles=static_obstacles, w_static=self.w_static,
                evasive=self.evasive)
            response.success = True
            response.status = status
            response.v, response.omega = float(u0[0]), float(u0[1])
            response.u_full_flat = u_full.reshape(-1).tolist()
            response.min_margin = float(margin)
            response.cost_ped = float("nan")

        elif request.controller_mode == "hard_v2":
            if v2_nmpc is None:
                response.success = False
                response.status = "hard_v2 richiede casadi"
                response.v = response.omega = 0.0
                response.u_full_flat = []
                response.min_margin = response.cost_ped = float("nan")
                return response
            obstacles = build_obstacles_hard(preds, positions_now, robot_xy, q_by_k,
                                                self.k_neigh, self.cutoff_m, n_mpc,
                                                clearance_l=self.clearance_l, q_by_id=q_by_id)
            obstacles += build_unpredicted_hard(unpred, robot_xy, self.cutoff_m, n_mpc,
                                                clearance_l=self.clearance_l, current=unpred_now)
            statics = [(o["xy"][0], o["xy"][1], o["d_safe"]) for o in static_obstacles]
            u0, ok, u_full, margin, status = self.v2_solve(
                x0, float(request.robot_v), float(request.robot_omega), goal, obstacles, n_mpc,
                request.v_max, request.omega_max, u_warm=u_warm, statics=statics)
            response.success = True
            response.status = status
            response.v, response.omega = float(u0[0]), float(u0[1])
            response.u_full_flat = u_full.reshape(-1).tolist()
            response.min_margin = float(margin)
            response.cost_ped = float("nan")

        elif request.controller_mode == "soft":
            obstacles = build_obstacles_soft(preds, positions_now, robot_xy, q_by_k,
                                                self.beta, self.gamma_exp, self.a_gauss,
                                                self.k_neigh, self.cutoff_m, n_mpc, q_by_id=q_by_id)
            static_obstacles = static_obstacles + unpredicted_as_static(
                unpred, robot_xy, self.cutoff_m, clearance_l=self.clearance_l)
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
