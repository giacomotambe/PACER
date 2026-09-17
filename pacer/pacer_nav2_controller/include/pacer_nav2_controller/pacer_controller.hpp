// Plugin nav2_core::Controller — integrazione DIRETTA con Nav2
// (§Sim2Real "tightly coupled": eredita lifecycle/recovery di Nav2, il
// local costmap resta interno invece che un topic esterno da tenere
// sincronizzato). Nav2 gestisce gli ostacoli STATICI (via costmap_ros_,
// interrogato direttamente qui — nessun vincolo rigido su di essi, solo un
// costo soft, §Sim2Real §2.3: il costmap non e' un vincolo di disuguaglianza
// liscio come l'ellisse di Mahalanobis); i pedoni DINAMICI restano
// competenza del modulo TUPAC del progetto (pacer_predictor +
// pacer_calibration), qui letti da due topic e passati al vero solver NMPC
// — che resta in Python (pacer_core, stesso codice validato in
// simulazione) — tramite il servizio pacer_msgs/ComputeMpcCommand servito
// da pacer_mpc_solver. Questo plugin e' quindi un PONTE, non reimplementa
// la matematica dell'NMPC in C++.
#ifndef PACER_NAV2_CONTROLLER__PACER_CONTROLLER_HPP_
#define PACER_NAV2_CONTROLLER__PACER_CONTROLLER_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2_ros/buffer.h"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

#include "pacer_msgs/msg/pedestrian_prediction_array.hpp"
#include "pacer_msgs/msg/q_tupac_snapshot.hpp"
#include "pacer_msgs/msg/static_obstacle.hpp"
#include "pacer_msgs/srv/compute_mpc_command.hpp"

namespace pacer_nav2_controller
{

class PacerController : public nav2_core::Controller
{
public:
  PacerController() = default;
  ~PacerController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

protected:
  geometry_msgs::msg::PoseStamped pickLocalGoal(const geometry_msgs::msg::PoseStamped & robot_pose);
  std::vector<pacer_msgs::msg::StaticObstacle> extractStaticObstacles(
    const geometry_msgs::msg::PoseStamped & robot_pose);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};
  std::string plugin_name_;
  rclcpp::Logger logger_{rclcpp::get_logger("PacerController")};

  // Nodo interno DEDICATO alla chiamata di servizio bloccante + alle due
  // sottoscrizioni pedoni/q_tupac: evita il deadlock che si avrebbe
  // chiamando spin_until_future_complete sullo stesso nodo/executor gia'
  // usato dal controller_server (pattern standard per client di servizio
  // sincroni dentro un plugin Nav2). Lo spin di questo nodo, fatto qui
  // dentro computeVelocityCommands mentre si attende la risposta del
  // servizio, serve ANCHE a processare i messaggi pedoni/q_tupac in coda.
  rclcpp::Node::SharedPtr client_node_;
  rclcpp::executors::SingleThreadedExecutor client_executor_;
  rclcpp::Client<pacer_msgs::srv::ComputeMpcCommand>::SharedPtr mpc_client_;
  rclcpp::Subscription<pacer_msgs::msg::PedestrianPredictionArray>::SharedPtr pred_sub_;
  rclcpp::Subscription<pacer_msgs::msg::QTupacSnapshot>::SharedPtr q_sub_;

  std::mutex state_mutex_;
  pacer_msgs::msg::PedestrianPredictionArray latest_predictions_;
  pacer_msgs::msg::QTupacSnapshot latest_q_tupac_;
  bool have_predictions_{false};
  bool have_q_tupac_{false};

  nav_msgs::msg::Path path_;
  std::vector<double> u_warm_;   // 2*n_mpc_, vuoto = nessun warm start disponibile

  // Parametri (§Sim2Real: stessi valori calibrati/verificati in simulazione,
  // esposti qui come parametri ROS2 invece che costanti hard-coded)
  std::string controller_mode_;      // "hard" o "soft"
  int n_mpc_{6};
  double v_max_{1.6};
  double omega_max_{2.0};
  double lookahead_distance_{1.5};
  double service_timeout_s_{0.35};   // < dt_out=0.4s (budget di ciclo, §A.5.2 nota MPC)
  double static_obstacle_radius_m_{5.0};
  double static_obstacle_cell_m_{0.3};
  int max_static_obstacles_{20};
  double static_obstacle_lethal_threshold_{253.0};
  double static_obstacle_d_safe_{0.5};   // raggio robot + margine di sicurezza residuo
};

}  // namespace pacer_nav2_controller

#endif  // PACER_NAV2_CONTROLLER__PACER_CONTROLLER_HPP_
