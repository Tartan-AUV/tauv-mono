from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share_dir = Path(get_package_share_directory("tauv_sim"))
    param_file = share_dir / "config" / "params.yaml"
    trajectory_file = share_dir / "config" / "trajectories" / "osprey_square.yaml"

    return LaunchDescription(
        [
            Node(
                package="tauv_sim",
                namespace="",
                executable="tauv_sim",
                name="tauv_sim",
                parameters=[str(param_file)],
                arguments=["--kinematic", str(trajectory_file)],
                output="screen",
            ),
        ]
    )
