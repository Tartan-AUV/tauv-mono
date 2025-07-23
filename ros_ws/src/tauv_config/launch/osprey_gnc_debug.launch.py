from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace, Node


def generate_launch_description():
    return LaunchDescription([
        GroupAction(actions=[
            PushRosNamespace("os"),

            # Depth estimator
            Node(
                package='tauv_common',
                executable='depth_estimator',
                name='depth_estimator',
                output='screen',
                remappings=[
                    ('depth_sensor_frame', 'vehicle/sensors/depth'),
                    ('depth', 'gnc/depth')
                ],
                parameters=[{
                    'surface_pressure': 101325.0,
                    'water_density': 997.0,
                    'gravity': 9.81,
                    'variance': 1.0e-4
                }]
            ),

            # Python State estimator EKF node
            Node(
                package='tauv_common',
                executable='state_estimator_ekf',
                name='state_estimator_ekf',
                output='screen',
                remappings=[
                    # Input topics - map to the topics published by the simulation
                    ('imu', 'vehicle/sensors/imu'),
                    ('depth', 'gnc/depth'),  # Now maps to the depth_estimator output
                    ('dvl', 'vehicle/sensors/dvl'),
                    # Output topics - keep default names
                    ('odom', 'gnc/odom')
                ],
                parameters=[{
                    'body_frame': 'os/body',
                    'depth_frame': 'os/depth',
                    'dvl_frame': 'os/dvl',
                    'imu_frame': 'os/body',
                    'initial_position_stddev_m': 0.01,
                    'initial_velocity_stddev_mps': 0.1,
                    'process_noise_density_pos_m_per_sqrt_s': 0.001,
                    'process_noise_density_vel_mps_per_sqrt_s': 0.001,
                    'g': 9.79596,
                    'history_length': 20
                }],
                # arguments=['--ros-args', '--log-level', 'debug']
            ),

            # Launch thruster manager node
            Node(
                package='tauv_common',
                executable='thruster_manager',
                name='thruster_manager'
            ),

            # Launch thruster controller node
            Node(
                package='tauv_common',
                executable='thruster_controller',
                name='thruster_controller'
            ),
        ])
    ])
