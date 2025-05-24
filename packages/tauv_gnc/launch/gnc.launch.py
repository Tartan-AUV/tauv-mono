from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    log_level = LaunchConfiguration('log_level')
    
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