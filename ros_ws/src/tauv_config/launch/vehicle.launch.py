from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():
    """Launch the RTVC (Real-Time Vehicle Controller) node with appropriate topic mappings."""

    foxglove_bridge_share = Path(get_package_share_directory('foxglove_bridge'))

    return LaunchDescription(
        [
            GroupAction(
                actions=[
                    # Use the same namespace as osprey_gnc_debug
                    PushRosNamespace("os"),
                    # RTVC node - interfaces with the real-time vehicle controller
                    Node(
                        package='tauv_vehicle',
                        executable='depth_actuators_i2c',
                        name='depth_actuators_i2c',
                        output='screen',
                        remappings=[
                            ('depth', 'vehicle/sensors/depth'),
                            ('thruster_setpoint', 'vehicle/actuators/thruster_setpoint'),
                        ],
                    ),
                    Node(
                        package='tauv_vehicle',
                        executable='waterlinked_driver',
                        name='waterlinked_driver',
                        output='screen',
                        remappings=[('dvl_frame', 'vehicle/sensors/dvl')],
                    ),
                    # Xsens MTi IMU node
                    Node(
                        package='xsens_mti_ros2_driver',
                        executable='xsens_mti_node',
                        name='xsens_mti_node',
                        output='screen',
                        parameters=[
                            Path(
                                get_package_share_directory('xsens_mti_ros2_driver'),
                                'param',
                                'xsens_mti_node.yaml',
                            )
                        ],
                        remappings=[
                            # Main IMU data goes to the standard vehicle sensors location
                            ('/imu/data', 'vehicle/sensors/imu'),
                            # All other topics go to raw subdirectory
                            (
                                '/filter/free_acceleration',
                                'vehicle/sensors/imu/raw/filter/free_acceleration',
                            ),
                            ('/filter/positionlla', 'vehicle/sensors/imu/raw/filter/positionlla'),
                            ('/filter/quaternion', 'vehicle/sensors/imu/raw/filter/quaternion'),
                            ('/filter/euler', 'vehicle/sensors/imu/raw/filter/euler'),
                            ('/filter/twist', 'vehicle/sensors/imu/raw/filter/twist'),
                            ('/filter/velocity', 'vehicle/sensors/imu/raw/filter/velocity'),
                            ('/gnss', 'vehicle/sensors/imu/raw/gnss'),
                            ('/gnss_pose', 'vehicle/sensors/imu/raw/gnss_pose'),
                            ('/imu/acceleration', 'vehicle/sensors/imu/raw/imu/acceleration'),
                            (
                                '/imu/angular_velocity',
                                'vehicle/sensors/imu/raw/imu/angular_velocity',
                            ),
                            ('/imu/dq', 'vehicle/sensors/imu/raw/imu/dq'),
                            ('/imu/dv', 'vehicle/sensors/imu/raw/imu/dv'),
                            ('/imu/mag', 'vehicle/sensors/imu/raw/imu/mag'),
                            ('/imu/time_ref', 'vehicle/sensors/imu/raw/imu/time_ref'),
                            ('/imu/utctime', 'vehicle/sensors/imu/raw/imu/utctime'),
                            ('/imu/acceleration_hr', 'vehicle/sensors/imu/raw/imu/acceleration_hr'),
                            (
                                '/imu/angular_velocity_hr',
                                'vehicle/sensors/imu/raw/imu/angular_velocity_hr',
                            ),
                            ('/nmea', 'vehicle/sensors/imu/raw/nmea'),
                            ('/pressure', 'vehicle/sensors/imu/raw/pressure'),
                            ('/status', 'vehicle/sensors/imu/raw/status'),
                            ('/temperature', 'vehicle/sensors/imu/raw/temperature'),
                            ('/tf', 'vehicle/sensors/imu/raw/tf'),
                        ],
                    ),
                ]
            ),
            # Include the foxglove bridge launch file (in root namespace)
            IncludeLaunchDescription(
                XMLLaunchDescriptionSource(
                    [str(foxglove_bridge_share / 'launch' / 'foxglove_bridge_launch.xml')]
                ),
                launch_arguments={'port': '8765'}.items(),
            ),
        ]
    )
