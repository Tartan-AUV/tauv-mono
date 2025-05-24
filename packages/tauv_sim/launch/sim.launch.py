from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory
from os.path import join

def generate_launch_description():
    pkg_share = get_package_share_directory('tauv_sim')
    
    return LaunchDescription([
        GroupAction(actions=[
            PushRosNamespace("os"),
            Node(
                package='stonefish_ros2',
                executable='stonefish_simulator',
                name='stonefish_simulator',
                arguments=[
                    join(pkg_share, 'data'),
                    join(pkg_share, 'scenarios/osprey_irvine.scn'),
                    '300',
                    '2000',
                    '2000',
                    'high'
                ],
                remappings=[
                    ('sim/imu', 'sensors/imu')
                ],
                output='screen'
            ),
            Node(
                package='tauv_sim',
                executable='sim_adapter',
                name='sim_adapter',
                output='screen',
                remappings=[
                    ('dvl', 'sensors/dvl'),
                    ('depth_sensor_frame', 'sensors/depth_sensor_frame')
                ],
            )
        ])
    ])
