from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from os.path import join

def generate_launch_description():
    pkg_share = get_package_share_directory('tauv_sim')
    
    return LaunchDescription([
        Node(
            package='stonefish_ros2',
            executable='stonefish_simulator',
            name='stonefish_simulator',
            arguments=[
                join(pkg_share, 'data'),
                join(pkg_share, 'scenarios/osprey_irvine.scn'),
                '300',
                '640',
                '480',
                'low'
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
                ('dvl', 'vehicle/sensors/dvl'),
                ('depth_sensor_frame', 'vehicle/sensors/depth')
            ],
        )
    ])
