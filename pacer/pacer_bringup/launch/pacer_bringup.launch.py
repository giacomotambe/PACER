"""Avvia i quattro nodi Python del modulo TUPAC (qualisys_bridge, predictor,
calibration, mpc_solver). NON avvia Nav2 (bt_navigator/controller_server/
planner_server/costmap): lo stack Nav2 va lanciato a parte (es. col proprio
launch/bringup standard), con nav2_params.yaml che include lo snippet in
config/nav2_params_snippet.yaml per puntare il controller_server al plugin
PacerController — mantenere i due lanci separati rende piu' facile
riavviare/debuggare il modulo TUPAC senza toccare Nav2 (§Sim2Real: nodi con
responsabilita' separate)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("pacer_bringup")
    default_params = os.path.join(pkg_share, "config", "pacer_params.yaml")

    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=default_params,
        description="Percorso di pacer_params.yaml (uno per tutti e 4 i nodi).")
    params_file = LaunchConfiguration("params_file")
    # rev. 2026-11-24: scelta del calibratore in un file a parte (config/calibrator.yaml), caricato dopo
    # params_file; calib_preset dal launch ha la precedenza su entrambi ("" = usa il file).
    calibrator_file_arg = DeclareLaunchArgument(
        "calibrator_file", default_value=os.path.join(pkg_share, "config", "calibrator.yaml"),
        description="File del calibratore (calib_preset / calib_method / calib_method_long / alpha_far).")
    calib_preset_arg = DeclareLaunchArgument(
        "calib_preset", default_value="",
        description="standard | conservative | legacy | static | custom; vuoto = quello di calibrator_file.")
    calibrator_file = LaunchConfiguration("calibrator_file")

    return LaunchDescription([
        params_file_arg,
        calibrator_file_arg,
        calib_preset_arg,
        Node(
            package="pacer_qualisys_bridge", executable="qualisys_bridge_node",
            name="pacer_qualisys_bridge", output="screen",
            parameters=[params_file],
        ),
        Node(
            package="pacer_predictor", executable="predictor_node",
            name="pacer_predictor", output="screen",
            parameters=[params_file],
        ),
        Node(
            package="pacer_calibration", executable="calibration_node",
            name="pacer_calibration", output="screen",
            parameters=[params_file, calibrator_file,
                        {"calib_preset_override": LaunchConfiguration("calib_preset")}],
        ),
        Node(
            package="pacer_mpc_solver", executable="mpc_solver_node",
            name="pacer_mpc_solver", output="screen",
            parameters=[params_file],
        ),
        # rev. 2026-11-17: filtro di sicurezza (passive safety) fra controller e base
        Node(
            package="pacer_mpc_solver", executable="safety_filter_node",
            name="pacer_safety_filter", output="screen",
            parameters=[params_file],
        ),
        # Toglie dalla point cloud del lidar i punti vicini ai pedoni
        # tracciati da Qualisys PRIMA che arrivi alla costmap di Nav2 (avviato
        # a parte, vedi docstring sopra) — il nav2_params.yaml del progetto va
        # puntato su /velodyne_points/pedestrian_filtered invece del topic
        # grezzo del Velodyne, vedi nota in config/nav2_params_snippet.yaml.
        Node(
            package="pacer_pedestrian_filter", executable="pedestrian_filter_node",
            name="pacer_pedestrian_filter", output="screen",
            parameters=[params_file],
        ),
    ])
