from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
        GroupAction(actions=[
            PushRosNamespace("os"),
            GroupAction(actions=[
                PushRosNamespace("gnc"),
                # Node(
                #     package='tauv_gnc',
                #     executable='state_estimator',
                #     name='state_estimator',
                #     remappings=[
                #         ('imu', '/os/sensors/imu'),
                #         ('dvl', '/os/sensors/dvl'),
                #         ('depth', '/os/sensors/depth')
                #     ],
                #     output='screen'
                # ),
                Node(
                    package='tauv_gnc',
                    executable='depth_estimator',
                    name='depth_estimator',
                    remappings=[
                        ('depth_sensor_frame', '/os/sensors/depth_sensor_frame'),
                        ('depth', '/os/sensors/depth')
                    ],
                    output='screen'
                )
            ])
        ])
    ]) 