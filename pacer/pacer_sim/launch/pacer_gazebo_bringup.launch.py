"""Punto di ingresso unico per testare pacer in simulazione:

  1. spawn_jo.launch.py       -> gz-sim + Jo (lidar standard) + bridge gz<->ROS2
  2. jo_navigation            -> stack Nav2 "local" (solo odom, no AMCL/mappa),
                                  controller_server puntato su PacerController
                                  via config/nav2_pacer_sim.yaml
  3. pacer_predictor/pacer_calibration/pacer_mpc_solver invariati (stessi
     nodi usati con hardware reale, pacer_bringup/config/pacer_params.yaml)
  4. sim_pedestrian_publisher al posto di pacer_qualisys_bridge — non c'e'
     motion capture in simulazione, i pedoni sono traiettorie scriptate
     (config/pacer_sim_pedestrians.yaml) pubblicate con lo STESSO contratto
     (stesso topic/messaggio) del bridge Qualisys reale.

L'EKF di jo_navigation (localization.launch.py) e' intenzionalmente escluso:
si aspetta glim_ros/odom_corrected (SLAM lidar reale, assente in questo
mondo di test) e pubblicherebbe un secondo odom->base_link TF in conflitto
con l'OdometryPublisher gia' nello xacro di pacer_sim (odometria ground
truth della simulazione, usata direttamente — vedi commenti in
urdf/jo_pacer_sim.urdf.xacro e config/nav2_pacer_sim.yaml)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pacer_sim_share = get_package_share_directory("pacer_sim")
    jo_navigation_share = get_package_share_directory("jo_navigation")
    pacer_bringup_share = get_package_share_directory("pacer_bringup")

    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="false",
        description="Avvia anche RViz (config/pacer_sim.rviz).")

    nav2_params_file = os.path.join(pacer_sim_share, "config", "nav2_pacer_sim.yaml")
    rviz_config_file = os.path.join(pacer_sim_share, "rviz", "pacer_sim.rviz")
    pedestrians_params_file = os.path.join(
        pacer_sim_share, "config", "pacer_sim_pedestrians.yaml")
    pacer_params_file = os.path.join(pacer_bringup_share, "config", "pacer_params.yaml")

    spawn_jo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pacer_sim_share, "launch", "spawn_jo.launch.py")),
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(jo_navigation_share, "launch", "navigation_local.launch.py")),
        launch_arguments={
            "use_sim_time": "true",
            "autostart": "true",
            "params_file": nav2_params_file,
            # jo_navigation's own RViz always assumes frame "map" (real
            # robot, AMCL/SLAM) and hardcodes that config path — unusable
            # here (no map frame, see README). Launched separately below,
            # with a config that only changes Fixed Frame to "odom".
            #
            # NOTE: this explicit "rviz": "false" here PERMANENTLY
            # overwrites the shared LaunchConfiguration("rviz") for the
            # rest of this launch tree — IncludeLaunchDescription's
            # launch_arguments mutate the parent context, they don't
            # create an isolated scope that gets restored afterward. Any
            # later use of LaunchConfiguration("rviz") in THIS file (e.g.
            # an IfCondition on our own rviz2 node below) would therefore
            # always see "false", regardless of what the user actually
            # passed. Resolved with an OpaqueFunction below, which reads
            # the value BEFORE this include runs.
            "rviz": "false",
        }.items(),
    )

    def _launch_rviz_if_enabled(context, *args, **kwargs):
        # Resolved here, before the nav2 IncludeLaunchDescription above
        # overwrites the shared "rviz" LaunchConfiguration — see the note
        # on its "rviz": "false" override. Must run before `nav2` in the
        # LaunchDescription below for that ordering to hold.
        if LaunchConfiguration("rviz").perform(context).lower() not in ("true", "1"):
            return []
        return [Node(
            package="rviz2", executable="rviz2", name="rviz2", output="screen",
            arguments=["-d", rviz_config_file],
            parameters=[{"use_sim_time": True}],
        )]

    rviz = OpaqueFunction(function=_launch_rviz_if_enabled)

    predictor_node = Node(
        package="pacer_predictor", executable="predictor_node",
        name="pacer_predictor", output="screen",
        parameters=[pacer_params_file, {"use_sim_time": True}],
    )
    calibration_node = Node(
        package="pacer_calibration", executable="calibration_node",
        name="pacer_calibration", output="screen",
        parameters=[pacer_params_file, {"use_sim_time": True}],
    )
    mpc_solver_node = Node(
        package="pacer_mpc_solver", executable="mpc_solver_node",
        name="pacer_mpc_solver", output="screen",
        parameters=[pacer_params_file, {"use_sim_time": True}],
    )
    sim_pedestrian_publisher = Node(
        package="pacer_sim", executable="sim_pedestrian_publisher",
        name="sim_pedestrian_publisher", output="screen",
        parameters=[pedestrians_params_file, {"use_sim_time": True}],
    )
    # Toglie dalla point cloud del lidar i punti vicini ai pedoni tracciati
    # PRIMA che arrivi alla costmap — vedi config/nav2_pacer_sim.yaml
    # (scan.topic puntato su /velodyne_points/pedestrian_filtered, non sul
    # /velodyne_points grezzo) e pacer_pedestrian_filter/README.md.
    pedestrian_filter_node = Node(
        package="pacer_pedestrian_filter", executable="pedestrian_filter_node",
        name="pacer_pedestrian_filter", output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        rviz_arg,
        rviz,   # deve restare PRIMA di `nav2` — vedi nota sopra
        spawn_jo,
        nav2,
        predictor_node, calibration_node, mpc_solver_node,
        sim_pedestrian_publisher,
        pedestrian_filter_node,
    ])
