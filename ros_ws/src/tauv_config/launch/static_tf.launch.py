from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get the URDF file path
    urdf_path = PathJoinSubstitution([FindPackageShare('tauv_config'), 'urdf', 'osprey.urdf'])

    # Load the URDF file content
    robot_description = Command(['cat ', urdf_path])

    return LaunchDescription(
        [
            # Robot state publisher for URDF transforms
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
                output='screen',
            ),
        ]
    )
