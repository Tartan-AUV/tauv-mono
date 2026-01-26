from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_share_dir = Path(get_package_share_directory("tauv_sim"))
    sim_param_file = sim_share_dir / "config" / "params.yaml"
    trajectory_file = sim_share_dir / "config" / "trajectories" / "osprey_square.yaml"

    common_share_dir = Path(get_package_share_directory("tauv_common"))
    common_ekf_file = common_share_dir / "config" / "ekf.yaml"

    # Timestamped bag name
    timestamp = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    bag_name = f"sim_{timestamp}"
    common_ekf_record_file = (
        Path("src") / "tauv_common" / "odometry_visualization" / "rosbags" / bag_name
    )
    print(f"Recording EKF data to: {common_ekf_record_file}")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'record', default_value='true', description='Enable rosbag recording'
            ),
            Node(
                package="tauv_sim",
                executable="tauv_sim",
                name="tauv_sim",
                parameters=[str(sim_param_file)],
                arguments=["--kinematic", str(trajectory_file)],
                output="screen",
            ),
            Node(
                package="tauv_common",
                executable="depth_converter",
                name="depth_converter",
                output="screen",
            ),
            Node(
                package="tauv_common",
                executable="dvl_converter",
                name="dvl_converter",
                output="screen",
            ),
            Node(
                package="robot_localization",
                executable="ekf_node",
                name="ekf_filter_node",
                parameters=[str(common_ekf_file)],
                output="screen",
            ),
            ExecuteProcess(
                condition=IfCondition(LaunchConfiguration('record')),
                cmd=[
                    'ros2',
                    'bag',
                    'record',
                    '/odometry/filtered',
                    '-o',
                    str(common_ekf_record_file),
                ],
                output='screen',
            ),
        ]
    )
