#include "pacer_nav2_controller/pacer_controller.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

#include "nav2_util/node_utils.hpp"
#include "tf2/utils.h"

namespace pacer_nav2_controller
{

void PacerController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  auto node = node_.lock();
  plugin_name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  logger_ = node->get_logger();

  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".controller_mode", rclcpp::ParameterValue("soft"));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".n_mpc", rclcpp::ParameterValue(6));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".v_max", rclcpp::ParameterValue(1.6));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".omega_max", rclcpp::ParameterValue(2.0));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".lookahead_distance", rclcpp::ParameterValue(1.5));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".service_timeout_s", rclcpp::ParameterValue(0.35));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".static_obstacle_radius_m", rclcpp::ParameterValue(5.0));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".static_obstacle_cell_m", rclcpp::ParameterValue(0.3));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_static_obstacles", rclcpp::ParameterValue(20));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".static_obstacle_d_safe", rclcpp::ParameterValue(0.5));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".mpc_service_name",
    rclcpp::ParameterValue(std::string("/pacer/compute_mpc_command")));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".pedestrian_predictions_topic",
    rclcpp::ParameterValue(std::string("/pacer/pedestrian_predictions")));
  nav2_util::declare_parameter_if_not_declared(
    node, plugin_name_ + ".q_tupac_topic",
    rclcpp::ParameterValue(std::string("/pacer/q_tupac")));

  node->get_parameter(plugin_name_ + ".controller_mode", controller_mode_);
  node->get_parameter(plugin_name_ + ".n_mpc", n_mpc_);
  node->get_parameter(plugin_name_ + ".v_max", v_max_);
  node->get_parameter(plugin_name_ + ".omega_max", omega_max_);
  node->get_parameter(plugin_name_ + ".lookahead_distance", lookahead_distance_);
  node->get_parameter(plugin_name_ + ".service_timeout_s", service_timeout_s_);
  node->get_parameter(plugin_name_ + ".static_obstacle_radius_m", static_obstacle_radius_m_);
  node->get_parameter(plugin_name_ + ".static_obstacle_cell_m", static_obstacle_cell_m_);
  node->get_parameter(plugin_name_ + ".max_static_obstacles", max_static_obstacles_);
  node->get_parameter(plugin_name_ + ".static_obstacle_d_safe", static_obstacle_d_safe_);

  std::string mpc_service_name, pred_topic, q_topic;
  node->get_parameter(plugin_name_ + ".mpc_service_name", mpc_service_name);
  node->get_parameter(plugin_name_ + ".pedestrian_predictions_topic", pred_topic);
  node->get_parameter(plugin_name_ + ".q_tupac_topic", q_topic);

  // Nodo interno per il client di servizio bloccante + le due
  // sottoscrizioni pedoni/q_tupac (vedi commento nell'header: evita il
  // deadlock di spin_until_future_complete sull'executor del
  // controller_server).
  client_node_ = std::make_shared<rclcpp::Node>(plugin_name_ + "_pacer_client");
  client_executor_.add_node(client_node_);
  mpc_client_ = client_node_->create_client<pacer_msgs::srv::ComputeMpcCommand>(mpc_service_name);
  pred_sub_ = client_node_->create_subscription<pacer_msgs::msg::PedestrianPredictionArray>(
    pred_topic, rclcpp::SensorDataQoS(),
    [this](pacer_msgs::msg::PedestrianPredictionArray::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      latest_predictions_ = *msg;
      have_predictions_ = true;
    });
  q_sub_ = client_node_->create_subscription<pacer_msgs::msg::QTupacSnapshot>(
    q_topic, 10,
    [this](pacer_msgs::msg::QTupacSnapshot::SharedPtr msg) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      latest_q_tupac_ = *msg;
      have_q_tupac_ = true;
    });

  RCLCPP_INFO(
    logger_, "PacerController configurato: mode=%s, n_mpc=%d, servizio=%s",
    controller_mode_.c_str(), n_mpc_, mpc_service_name.c_str());
}

void PacerController::cleanup()
{
  RCLCPP_INFO(logger_, "PacerController cleanup");
  client_executor_.remove_node(client_node_);
  mpc_client_.reset();
  pred_sub_.reset();
  q_sub_.reset();
  client_node_.reset();
}

void PacerController::activate() { RCLCPP_INFO(logger_, "PacerController activate"); }
void PacerController::deactivate() { RCLCPP_INFO(logger_, "PacerController deactivate"); }

void PacerController::setPlan(const nav_msgs::msg::Path & path)
{
  path_ = path;
  u_warm_.clear();   // nuovo piano -> nessun warm start valido dal ciclo precedente
}

void PacerController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (percentage) {
    v_max_ = v_max_ * speed_limit / 100.0;
  } else {
    v_max_ = speed_limit;
  }
}

geometry_msgs::msg::PoseStamped PacerController::pickLocalGoal(
  const geometry_msgs::msg::PoseStamped & robot_pose)
{
  // Punto di lookahead lungo il path globale di Nav2 (stessa idea di base
  // dei controller "pure pursuit" di Nav2): il primo punto del path a
  // distanza >= lookahead_distance_ dal robot, o l'ultimo se il path e'
  // piu' corto. L'NMPC (pacer_core) riceve quindi un goal LOCALE, non
  // l'obiettivo finale della missione — coerente con l'orizzonte breve
  // (N_mpc<=6 passi, 2.4s) gia' usato in simulazione.
  if (path_.poses.empty()) {
    return robot_pose;
  }
  for (const auto & ps : path_.poses) {
    double dx = ps.pose.position.x - robot_pose.pose.position.x;
    double dy = ps.pose.position.y - robot_pose.pose.position.y;
    if (std::hypot(dx, dy) >= lookahead_distance_) {
      return ps;
    }
  }
  return path_.poses.back();
}

std::vector<pacer_msgs::msg::StaticObstacle> PacerController::extractStaticObstacles(
  const geometry_msgs::msg::PoseStamped & robot_pose)
{
  // Celle "lethal"/inflated del local costmap entro static_obstacle_radius_m_
  // dal robot, clusterizzate su una griglia grossolana (static_obstacle_cell_m_)
  // per non passare centinaia di punti al solver Python ad ogni ciclo —
  // §Sim2Real §2.3: trattate come costo soft (mai vincolo rigido, riservato
  // ai pedoni TUPAC).
  std::vector<pacer_msgs::msg::StaticObstacle> obstacles;
  costmap_ = costmap_ros_->getCostmap();
  if (!costmap_) {
    return obstacles;
  }
  double rx = robot_pose.pose.position.x, ry = robot_pose.pose.position.y;
  double res = costmap_->getResolution();
  unsigned int mx0, my0;
  if (!costmap_->worldToMap(rx, ry, mx0, my0)) {
    return obstacles;
  }
  int cell_radius = static_cast<int>(std::round(static_obstacle_radius_m_ / res));
  int bin_cells = std::max(1, static_cast<int>(std::round(static_obstacle_cell_m_ / res)));

  std::map<std::pair<int, int>, std::pair<double, int>> bins;   // (bin_x,bin_y) -> (somma_costo, conteggio)
  std::map<std::pair<int, int>, std::pair<double, double>> centroids;

  int x_min = std::max(0, static_cast<int>(mx0) - cell_radius);
  int x_max = std::min(static_cast<int>(costmap_->getSizeInCellsX()) - 1,
                          static_cast<int>(mx0) + cell_radius);
  int y_min = std::max(0, static_cast<int>(my0) - cell_radius);
  int y_max = std::min(static_cast<int>(costmap_->getSizeInCellsY()) - 1,
                          static_cast<int>(my0) + cell_radius);

  for (int my = y_min; my <= y_max; ++my) {
    for (int mx = x_min; mx <= x_max; ++mx) {
      unsigned char cost = costmap_->getCost(mx, my);
      if (cost < static_obstacle_lethal_threshold_) {
        continue;
      }
      std::pair<int, int> bin{mx / bin_cells, my / bin_cells};
      double wx, wy;
      costmap_->mapToWorld(mx, my, wx, wy);
      auto & c = centroids[bin];
      auto & b = bins[bin];
      c.first += wx; c.second += wy;
      b.first += 1.0; b.second += 1;
    }
  }

  std::vector<std::pair<double, std::pair<double, double>>> candidates;   // (dist, xy)
  for (auto & kv : centroids) {
    int n = bins[kv.first].second;
    double cx = kv.second.first / n, cy = kv.second.second / n;
    double d = std::hypot(cx - rx, cy - ry);
    candidates.emplace_back(d, std::make_pair(cx, cy));
  }
  std::sort(candidates.begin(), candidates.end(),
             [](const auto & a, const auto & b) { return a.first < b.first; });
  int n_take = std::min<int>(max_static_obstacles_, candidates.size());
  for (int i = 0; i < n_take; ++i) {
    pacer_msgs::msg::StaticObstacle obs;
    obs.xy.x = candidates[i].second.first;
    obs.xy.y = candidates[i].second.second;
    obs.d_safe = static_obstacle_d_safe_;
    obstacles.push_back(obs);
  }
  return obstacles;
}

geometry_msgs::msg::TwistStamped PacerController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & /*velocity*/,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  geometry_msgs::msg::TwistStamped cmd;
  cmd.header.stamp = pose.header.stamp;
  cmd.header.frame_id = pose.header.frame_id;
  cmd.twist.linear.x = 0.0;
  cmd.twist.angular.z = 0.0;

  if (!mpc_client_->service_is_ready()) {
    RCLCPP_WARN_THROTTLE(
      logger_, *client_node_->get_clock(), 2000,
      "servizio pacer_mpc_solver non disponibile: comando zero (Nav2 gestira' il "
      "recovery). Verifica che il nodo pacer_mpc_solver sia avviato.");
    return cmd;
  }

  auto local_goal = pickLocalGoal(pose);

  auto request = std::make_shared<pacer_msgs::srv::ComputeMpcCommand::Request>();
  request->controller_mode = controller_mode_;
  request->robot_x = pose.pose.position.x;
  request->robot_y = pose.pose.position.y;
  request->robot_theta = tf2::getYaw(pose.pose.orientation);
  request->goal_x = local_goal.pose.position.x;
  request->goal_y = local_goal.pose.position.y;
  request->n_mpc = n_mpc_;
  request->v_max = v_max_;
  request->omega_max = omega_max_;
  request->u_warm_flat = u_warm_;
  request->static_obstacles = extractStaticObstacles(pose);

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (have_predictions_) {
      request->pedestrians = latest_predictions_;
    }
    if (have_q_tupac_) {
      request->q_tupac = latest_q_tupac_;
    }
  }

  auto future = mpc_client_->async_send_request(request);
  auto timeout = std::chrono::duration<double>(service_timeout_s_);
  // Spin del nodo client DEDICATO (non del controller_server, vedi header):
  // processa anche i messaggi pedoni/q_tupac in coda mentre attende.
  auto rc = client_executor_.spin_until_future_complete(future, timeout);
  if (rc != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_WARN(
      logger_, "pacer_mpc_solver non ha risposto entro %.2fs: comando zero.",
      service_timeout_s_);
    u_warm_.clear();
    return cmd;
  }

  auto response = future.get();
  if (!response->success) {
    RCLCPP_WARN(logger_, "ComputeMpcCommand fallito: %s", response->status.c_str());
    u_warm_.clear();
    return cmd;
  }

  cmd.twist.linear.x = response->v;
  cmd.twist.angular.z = response->omega;

  // Warm start per il prossimo ciclo — stessa logica di
  // pacer_core.unicycle_nmpc[_soft].solve_step*: soft aggiorna SEMPRE,
  // hard solo se lo status e' "optimal" (altrimenti la sequenza
  // pianificata potrebbe non essere una buona base di partenza).
  bool keep_warm = (controller_mode_ == "soft") || (response->status == "optimal");
  if (keep_warm && static_cast<int>(response->u_full_flat.size()) == 2 * n_mpc_) {
    std::vector<double> shifted(2 * n_mpc_);
    for (int i = 0; i < n_mpc_ - 1; ++i) {
      shifted[2 * i] = response->u_full_flat[2 * (i + 1)];
      shifted[2 * i + 1] = response->u_full_flat[2 * (i + 1) + 1];
    }
    shifted[2 * (n_mpc_ - 1)] = response->u_full_flat[2 * (n_mpc_ - 1)];
    shifted[2 * (n_mpc_ - 1) + 1] = response->u_full_flat[2 * (n_mpc_ - 1) + 1];
    u_warm_ = shifted;
  } else {
    u_warm_.clear();
  }

  return cmd;
}

}  // namespace pacer_nav2_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(pacer_nav2_controller::PacerController, nav2_core::Controller)
