from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from os.path import join

def generate_launch_description():
    pkg_share = get_package_share_directory('tauv_sim')
    
    return LaunchDescription([
        Node(
            package='stonefish_ros2',
            executable='tauv_sim',
            name='tauv_sim',
            arguments=[
                join(pkg_share, 'data'),
                join(pkg_share, 'scenarios/osprey_irvine.scn'),
                '300',
                '640',
                '480',
                'low'
            ],
            remappings=[
                ('sim/imu', 'sensors/imu'),
                # Publish sim DVL/pressure directly to vehicle topics
                ('sim/dvl', 'vehicle/sensors/dvl'),
                ('sim/pressure', 'vehicle/sensors/depth'),
                # Accept TAUV thruster setpoints on vehicle topic
                ('sim/thruster_setpoint', 'vehicle/actuators/thruster_setpoint')
            ],
            output='screen'
        ),
    ])
