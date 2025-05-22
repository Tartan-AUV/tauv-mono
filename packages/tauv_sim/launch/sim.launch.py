from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
        GroupAction(actions=[
            PushRosNamespace("os"),
            Node(
                package='stonefish_ros2',
                executable='stonefish_simulator',
                name='stonefish_simulator',
                arguments=[
                    './src/packages/tauv_sim/data',
                    './src/packages/tauv_sim/scenarios/osprey_irvine.scn',
                    '300',
                    '1280',
                    '800',
                    'low'
                ],
                remappings=[
                    ('sim/imu', 'imu')
                ],
                output='screen'
            ),
            Node(
                package='tauv_sim',
                executable='sim_adapter',
                name='sim_adapter',
                output='screen'
            )
        ])
    ])
