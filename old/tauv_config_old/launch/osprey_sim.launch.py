from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    # Get the package share directories
    tauv_config_share = get_package_share_directory('tauv_config')
    tauv_sim_share = get_package_share_directory('tauv_sim')

    return LaunchDescription(
        [
            GroupAction(
                actions=[
                    PushRosNamespace("os"),
                    # Include the simulation launch file
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            [join(tauv_sim_share, 'launch', 'sim.launch.py')]
                        )
                    ),
                    # Include the static TF publisher launch file
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            [join(tauv_config_share, 'launch', 'static_tf.launch.py')]
                        )
                    ),
                    # Include the state estimator launch file
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            [join(tauv_config_share, 'launch', 'state_estimator.launch.py')]
                        )
                    ),
                    # Launch thruster manager node
                    Node(
                        package='tauv_common',
                        executable='thruster_manager',
                        name='thruster_manager',
                    ),
                    # Launch thruster controller node
                    Node(
                        package='tauv_vehicle',
                        executable='thruster_controller',
                        name='thruster_controller',
                    ),
                ]
            )
        ]
    )
