import os

import launch_ros
from launch_ros.actions import Node
import xacro
from launch.conditions import IfCondition

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():

    # Check if we're told to use sim time
    use_sim_time = LaunchConfiguration('use_sim_time')

    rviz_arg = DeclareLaunchArgument('use_rviz', default_value='false', description='Whether to launch RViz')

    rviz_config = os.path.join(
        get_package_share_directory('jo_description'),
        'rviz',
        'display.rviz'
        )
    

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    # Process the URDF file
    pkg_path = os.path.join(get_package_share_directory('jo_description'))
    xacro_file = os.path.join(pkg_path,'urdf','jo_main.urdf.xacro')
    robot_description_config = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )    
    # Create a robot_state_publisher node
    params = {'robot_description': robot_description_config, 'use_sim_time': use_sim_time}
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )


    # Launch!
    return LaunchDescription([
        rviz_arg,
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use sim time if true'),
        DeclareLaunchArgument(
            'use_ros2_control',
            default_value='true',
            description='Use ros2_control if true'),

        node_robot_state_publisher,
        rviz,
        
    ])