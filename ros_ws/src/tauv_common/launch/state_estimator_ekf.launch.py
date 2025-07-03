from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'body_frame',
            default_value='body',
            description='Body frame name'
        ),
        DeclareLaunchArgument(
            'depth_frame',
            default_value='depth',
            description='Depth sensor frame name'
        ),
        DeclareLaunchArgument(
            'dvl_frame',
            default_value='dvl',
            description='DVL frame name'
        ),
        DeclareLaunchArgument(
            'initial_position_stddev_m',
            default_value='0.01',
            description='Initial position standard deviation in meters'
        ),
        DeclareLaunchArgument(
            'initial_velocity_stddev_mps',
            default_value='0.1',
            description='Initial velocity standard deviation in m/s'
        ),
        DeclareLaunchArgument(
            'process_noise_density_pos_m_per_sqrt_s',
            default_value='0.001',
            description='Process noise density for position'
        ),
        DeclareLaunchArgument(
            'process_noise_density_vel_mps_per_sqrt_s',
            default_value='0.001',
            description='Process noise density for velocity'
        ),
        DeclareLaunchArgument(
            'g',
            default_value='9.79596',
            description='Gravitational acceleration'
        ),
        DeclareLaunchArgument(
            'history_length',
            default_value='20',
            description='Maximum history length'
        ),
        
        # State estimator EKF node
        Node(
            package='tauv_common',
            executable='state_estimator_ekf',
            name='state_estimator_ekf',
            output='screen',
            parameters=[{
                'body_frame': LaunchConfiguration('body_frame'),
                'depth_frame': LaunchConfiguration('depth_frame'),
                'dvl_frame': LaunchConfiguration('dvl_frame'),
                'initial_position_stddev_m': LaunchConfiguration('initial_position_stddev_m'),
                'initial_velocity_stddev_mps': LaunchConfiguration('initial_velocity_stddev_mps'),
                'process_noise_density_pos_m_per_sqrt_s': LaunchConfiguration('process_noise_density_pos_m_per_sqrt_s'),
                'process_noise_density_vel_mps_per_sqrt_s': LaunchConfiguration('process_noise_density_vel_mps_per_sqrt_s'),
                'g': LaunchConfiguration('g'),
                'history_length': LaunchConfiguration('history_length'),
            }],
            remappings=[
                ('imu', 'sensors/imu'),
                ('depth', 'depth'),
                ('dvl', 'sensors/dvl'),
                ('odom', 'odom')
            ]
        )
    ]) 