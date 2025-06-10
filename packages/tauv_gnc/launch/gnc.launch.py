from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command

def generate_launch_description():
    log_level = LaunchConfiguration('log_level')
    
    # Get path to URDF file
    tauv_config_path = get_package_share_directory('tauv_config')
    urdf_path = os.path.join(tauv_config_path, 'urdf', 'osprey.urdf')
    
    # Read URDF file content
    with open(urdf_path, 'r') as infp:
        robot_desc = infp.read()
    
    return LaunchDescription([
        DeclareLaunchArgument(
            "log_level",
            default_value=["debug"],
        ),
        GroupAction(actions=[
            PushRosNamespace("os"),
            GroupAction(actions=[
                PushRosNamespace("gnc"),
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    parameters=[{
                        'robot_description': robot_desc,
                        'tf_prefix': 'os'
                    }],
                    arguments=['--ros-args', '--log-level', log_level],
                    output='screen'
                ),
                Node(
                    package='tauv_gnc',
                    executable='state_estimator',
                    name='state_estimator',
                    remappings=[
                        ('imu', '/os/sensors/imu'),
                        ('dvl', '/os/sensors/dvl'),
                        ('depth', '/os/sensors/depth')
                    ],
                    arguments=['--ros-args', '--log-level', log_level],
                    output='screen'
                ),
                Node(
                    package='tauv_gnc',
                    executable='depth_estimator',
                    name='depth_estimator',
                    remappings=[
                        ('depth_sensor_frame', '/os/sensors/depth_sensor_frame'),
                        ('depth', '/os/sensors/depth')
                    ],
                    # arguments=['--ros-args', '--log-level', log_level],
                    output='screen'
                )
            ])
        ])
    ]) 