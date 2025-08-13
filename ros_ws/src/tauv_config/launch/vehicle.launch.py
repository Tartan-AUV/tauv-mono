from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace, Node


def generate_launch_description():
    """Launch the RTVC (Real-Time Vehicle Controller) node with appropriate topic mappings."""
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
        ])
    ])