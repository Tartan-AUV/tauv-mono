from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace, Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """Launch the RTVC (Real-Time Vehicle Controller) node with appropriate topic mappings."""

    foxglove_bridge_share = Path(get_package_share_directory('foxglove_bridge'))

    return LaunchDescription([
        GroupAction(actions=[
            # Use the same namespace as osprey_gnc_debug
            PushRosNamespace("os"),

            # RTVC node - interfaces with the real-time vehicle controller
            Node(
                package='tauv_vehicle',
                executable='rtvc',
                name='rtvc_driver',
                output='screen',
                remappings=[
                    ('imu', 'vehicle/sensors/imu'),
                    ('depth', 'vehicle/sensors/depth'),
                    ('temperature', 'vehicle/sensors/temperature'),
                    ('pressure', 'vehicle/sensors/pressure'),
                    ('esc_telemetry', 'vehicle/esc_telemetry'),
                    ('thruster_setpoint', 'vehicle/actuators/thruster_setpoint'),
                ],
                parameters=[{
                    # Add any RTVC-specific parameters here if needed
                    # For example, network configuration, update rates, etc.
                }]
            ),

            Node(
                package='tauv_vehicle',
                executable='waterlinked_driver',
                name='waterlinked_driver',
                output='screen',
                remappings=[
                    ('dvl_frame', 'vehicle/sensors/dvl')
                ]
            )
        ]),
        # Include the foxglove bridge launch file (in root namespace)
        IncludeLaunchDescription(
            XMLLaunchDescriptionSource([
                str(foxglove_bridge_share / 'launch' / 'foxglove_bridge_launch.xml')
            ]),
            launch_arguments={'port': '8765'}.items()
        ),
    ])