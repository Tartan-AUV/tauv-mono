from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    param_file = Path(get_package_share_directory("tauv_sim")) / "config" / "params.yaml"

    print(param_file)

    return LaunchDescription(
        [
            Node(
                package="tauv_sim",
                namespace="",
                executable="tauv_sim",
                name="tauv_sim",
                parameters=[str(param_file)],
                output="screen",
            ),
        ]
    )
