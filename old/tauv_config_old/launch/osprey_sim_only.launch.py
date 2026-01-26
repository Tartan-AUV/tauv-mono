from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import PushRosNamespace
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource


def generate_launch_description():
    # Get the package share directories
    tauv_config_share = Path(get_package_share_directory('tauv_config'))
    tauv_sim_share = Path(get_package_share_directory('tauv_sim'))
    foxglove_bridge_share = Path(get_package_share_directory('foxglove_bridge'))

    return LaunchDescription(
        [
            GroupAction(
                actions=[
                    PushRosNamespace("os"),
                    # Include the simulation launch file
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            [str(tauv_sim_share / 'launch' / 'sim.launch.py')]
                        )
                    ),
                    # Include the static TF publisher launch file
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            [str(tauv_config_share / 'launch' / 'static_tf.launch.py')]
                        )
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
