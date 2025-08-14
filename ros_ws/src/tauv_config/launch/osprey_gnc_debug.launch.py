from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace, Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    tauv_config_share = Path(get_package_share_directory('tauv_config'))

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
                    ('odom', 'gnc/odom'),
                    # Map navigation_state to the gnc namespace for controllers
                    ('navigation_state', 'gnc/navigation_state')
                ],
                parameters=[{
                    'body_frame': 'os/body',
                    'depth_frame': 'os/depth',
                    'dvl_frame': 'os/dvl',
                    'imu_frame': 'os/imu',
                    'initial_position_stddev_m': 0.01,
                    'initial_velocity_stddev_mps': 0.1,
                    'process_noise_density_pos_m_per_sqrt_s': 0.001,
                    'process_noise_density_vel_mps_per_sqrt_s': 0.001,
                    'g': 9.79596,
                    'history_length': 20
                }],
                # arguments=['--ros-args', '--log-level', 'debug']
            ),

            # Commander node - converts velocity/attitude commands to acceleration commands
            Node(
                package='tauv_common',
                executable='commander',
                name='commander',
                output='screen',
                # No remapping needed - topics are already in gnc namespace
                parameters=[{
                    'kp_velocity': 2.0,
                    'kd_velocity': 0.1,
                    'kp_attitude': 1.5,
                    'kd_attitude': 0.3,
                    'max_linear_accel': 2.0,
                    'max_angular_accel': 1.0
                }]
            ),

            # INDI Controller node - converts acceleration commands to wrench commands
            Node(
                package='tauv_common',
                executable='indi_controller',
                name='indi_controller',
                output='screen',
                # No remapping needed - topics are already in gnc namespace
                parameters=[{
                    'accel_filter_alpha': 0.8,
                    'max_force': 100.0,
                    'max_torque': 50.0
                }]
            ),

            # Launch thruster manager node
            Node(
                package='tauv_common',
                executable='thruster_manager',
                name='thruster_manager',
                remappings=[
                    ('target_wrench', 'gnc/target_wrench'),
                    ('target_thrust', 'gnc/target_thrust'),
                ]
            ),

            # Launch thruster controller node
            Node(
                package='tauv_common',
                executable='thruster_controller',
                name='thruster_controller',
                remappings=[
                    ('target_thrust', 'gnc/target_thrust'),
                    ('thruster_setpoint', 'vehicle/actuators/thruster_setpoint'),
                ]
            ),
        ]),

        # Include the static TF publisher launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                str(tauv_config_share / 'launch' / 'static_tf.launch.py')
            ])
        ),
    ])
