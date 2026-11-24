"""Avvia i nodi del modulo PACER/TUPAC sul robot reale: qualisys_bridge, predictor,
calibration, mpc_solver, safety_filter e (opzionale) pedestrian_filter.

NON avvia Nav2 (bt_navigator/controller_server/planner_server/costmap) ne' il driver
Qualisys: vanno lanciati a parte (Nav2 col bringup del robot, con nav2_params.yaml che
include config/nav2_params_snippet.yaml; il driver con
`ros2 launch qualisys_driver qualisys.launch.py`). Lanci separati rendono piu' facile
riavviare/debuggare il modulo TUPAC senza toccare il resto.

Uso tipico (rev. 2026-11-24, versione pronta all'uso):
    ros2 launch pacer_bringup pacer_bringup.launch.py robot_body:=omnicar run_name:=auto

Argomenti:
    params_file        pacer_params.yaml (default: quello del pacchetto)
    calibrator_file    config/calibrator.yaml (calib_preset & co.)
    calib_preset       standard | conservative | legacy | static | custom ("" = dal file)
    robot_body         nome del rigid body del robot in QTM ("" = quello di params_file)
    model              pesi del predittore ("" = model_path di params_file, /models/predictor_sim.pt)
    calibration        D_cal .npz ("" = calibration_path di params_file, /models/cal_sim.npz);
                       dopo tools/make_dcal_real.sh: calibration:=/models/cal_real.npz
    run_name           "" = nessun log; "auto" = data e ora; altro = nome della prova.
                       I CSV (tracce mocap, score, stato MPC, filtro di sicurezza) vanno in
                       <log_dir>/<run_name>/, cioe' ./bags/runs/<run_name>/ sull'host.
    log_dir            default /home/ros/bags/runs
    pedestrian_filter  true | false (filtro della nuvola del lidar per la costmap di Nav2)
"""
import datetime
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)   # noqa: E731
    params_file, calibrator_file = lc("params_file"), lc("calibrator_file")
    robot_body, run_name = lc("robot_body"), lc("run_name")

    over = {n: {} for n in ("bridge", "pred", "calib", "mpc", "safety")}
    if lc("model"):
        over["pred"]["model_path"] = lc("model")
    if lc("calibration"):
        over["calib"]["calibration_path"] = lc("calibration")
    actions = []
    if robot_body:
        over["bridge"]["robot_body_name"] = robot_body
    if run_name:
        if run_name == "auto":
            run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(lc("log_dir"), run_name)
        os.makedirs(run_dir, exist_ok=True)
        over["bridge"]["tracks_log_csv"] = os.path.join(run_dir, "tracks.csv")
        over["calib"]["score_log_csv"] = os.path.join(run_dir, "scores.csv")
        over["mpc"]["status_log_csv"] = os.path.join(run_dir, "mpc_status.csv")
        over["safety"]["log_csv"] = os.path.join(run_dir, "safety.csv")
        actions.append(LogInfo(msg=f"[pacer] log della prova in {run_dir}"))

    def p(key, *extra):
        return [params_file, *extra] + ([over[key]] if over.get(key) else [])

    actions += [
        Node(package="pacer_qualisys_bridge", executable="qualisys_bridge_node",
             name="pacer_qualisys_bridge", output="screen", parameters=p("bridge")),
        Node(package="pacer_predictor", executable="predictor_node",
             name="pacer_predictor", output="screen", parameters=p("pred")),
        Node(package="pacer_calibration", executable="calibration_node",
             name="pacer_calibration", output="screen",
             parameters=p("calib", calibrator_file, {"calib_preset_override": lc("calib_preset")})),
        Node(package="pacer_mpc_solver", executable="mpc_solver_node",
             name="pacer_mpc_solver", output="screen", parameters=p("mpc")),
        # filtro di sicurezza (passive safety) fra controller e base
        Node(package="pacer_mpc_solver", executable="safety_filter_node",
             name="pacer_safety_filter", output="screen", parameters=p("safety")),
        # Toglie dalla nuvola del lidar i punti vicini ai pedoni tracciati, prima della costmap
        # di Nav2 (vedi nota in config/nav2_params_snippet.yaml). Senza lidar: pedestrian_filter:=false.
        Node(package="pacer_pedestrian_filter", executable="pedestrian_filter_node",
             name="pacer_pedestrian_filter", output="screen", parameters=[params_file],
             condition=IfCondition(LaunchConfiguration("pedestrian_filter"))),
    ]
    return actions


def generate_launch_description():
    share = get_package_share_directory("pacer_bringup")
    args = [
        DeclareLaunchArgument("params_file", default_value=os.path.join(share, "config", "pacer_params.yaml"),
                              description="pacer_params.yaml (uno per tutti i nodi)."),
        DeclareLaunchArgument("calibrator_file", default_value=os.path.join(share, "config", "calibrator.yaml"),
                              description="File del calibratore (calib_preset / calib_method / ...)."),
        DeclareLaunchArgument("calib_preset", default_value="",
                              description="standard | conservative | legacy | static | custom; vuoto = dal file."),
        DeclareLaunchArgument("robot_body", default_value="",
                              description="Rigid body del robot in QTM (vuoto = robot_body_name di params_file)."),
        DeclareLaunchArgument("model", default_value="",
                              description="Pesi del predittore (vuoto = model_path di params_file)."),
        DeclareLaunchArgument("calibration", default_value="",
                              description="D_cal .npz (vuoto = calibration_path di params_file)."),
        DeclareLaunchArgument("run_name", default_value="",
                              description="Nome della prova per i log CSV ('' = nessun log, 'auto' = data e ora)."),
        DeclareLaunchArgument("log_dir", default_value="/home/ros/bags/runs",
                              description="Cartella dei log (montata da ./bags sull'host)."),
        DeclareLaunchArgument("pedestrian_filter", default_value="true",
                              description="Avvia il filtro della nuvola del lidar (false se non c'e' lidar)."),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_nodes)])
