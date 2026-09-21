"""Avvia gz-sim con pacer_test_world, pubblica il robot_state_publisher del
robot Jo (urdf/jo_pacer_sim.urdf.xacro — jo_description reale + lidar
standard al posto di RGL + ruote virtuali DiffDrive al posto dei cingoli
TrackedVehicle, vedi commento nello xacro), lo spawna nel mondo e fa da
bridge gz<->ROS2 per i topic che servono a Nav2/pacer: clock, cmd_vel,
odom (dal plugin OdometryPublisher), tf, imu, velodyne_points.

Verificato per davvero contro un'istanza gz-sim in esecuzione (non solo
staticamente): topic names, tipi e subscriber controllati con `ign topic
-l` / `ign topic -i` a runtime — vedi i commenti sui singoli argomenti del
bridge sotto per cosa e' stato scoperto/corretto in questo modo."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pacer_sim_share = get_package_share_directory("pacer_sim")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    world_arg = DeclareLaunchArgument(
        "world", default_value=os.path.join(pacer_sim_share, "worlds", "pacer_test_world.sdf"),
        description="Percorso del mondo gz-sim da avviare.")
    spawn_x = DeclareLaunchArgument("spawn_x", default_value="0.0")
    spawn_y = DeclareLaunchArgument("spawn_y", default_value="0.0")
    spawn_z = DeclareLaunchArgument("spawn_z", default_value="0.5")
    spawn_yaw = DeclareLaunchArgument("spawn_yaw", default_value="0.0")

    xacro_file = os.path.join(pacer_sim_share, "urdf", "jo_pacer_sim.urdf.xacro")
    robot_description = ParameterValue(
        Command(["xacro ", xacro_file]), value_type=str)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": [LaunchConfiguration("world"), " -r"]}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        name="robot_state_publisher", output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    spawn_robot = Node(
        package="ros_gz_sim", executable="create",
        name="spawn_jo", output="screen",
        arguments=[
            "-name", "jo",
            "-topic", "robot_description",
            "-x", LaunchConfiguration("spawn_x"),
            "-y", LaunchConfiguration("spawn_y"),
            "-z", LaunchConfiguration("spawn_z"),
            "-Y", LaunchConfiguration("spawn_yaw"),
        ],
    )

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        name="pacer_sim_gz_bridge", output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # DiffDrive (unlike TrackedVehicle, see urdf/jo_pacer_sim.urdf.xacro)
            # genuinely honours <topic>cmd_vel</topic> — verify with
            # `ign topic -i -t /cmd_vel` after spawn that it shows up as a
            # subscriber, same check that caught TrackedVehicle silently
            # ignoring its own <topic> tag earlier.
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            # gz-sim's gpu_lidar auto-suffixes the point cloud onto
            # "<topic>/points" — "/velodyne_points" itself carries a
            # LaserScan, not a PointCloudPacked (confirmed with
            # `ign topic -l` / `ign topic -i` against a running sim:
            # /velodyne_points -> ignition.msgs.LaserScan,
            # /velodyne_points/points -> ignition.msgs.PointCloudPacked).
            # Bridged from the real topic, remapped back to
            # /velodyne_points on the ROS side so nav2_pacer_sim.yaml's
            # costmap "scan" source doesn't need to change.
            "/velodyne_points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        remappings=[
            ("/velodyne_points/points", "/velodyne_points"),
        ],
    )

    return LaunchDescription([
        world_arg, spawn_x, spawn_y, spawn_z, spawn_yaw,
        gz_sim, robot_state_publisher, spawn_robot, bridge,
    ])
