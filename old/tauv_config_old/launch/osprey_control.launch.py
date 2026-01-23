from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    tauv_config_share = Path(get_package_share_directory('tauv_config'))

    return LaunchDescription(
        [
            GroupAction(
                actions=[
                    PushRosNamespace("os"),
                    # Commander node - converts velocity/attitude commands to acceleration commands
                    # Node(
                    #     package='tauv_common',
                    #     executable='commander',
                    #     name='commander',
                    #     output='screen',
                    # ),
                    # # # INDI Controller node - converts acceleration commands to wrench commands
                    Node(
                        package='tauv_common',
                        executable='controller',
                        name='controller',
                        output='screen',
                    ),
                    # Launch thruster manager node
                    Node(
                        package='tauv_common',
                        executable='thruster_manager',
                        name='thruster_manager',
                        remappings=[
                            ('target_wrench', 'gnc/target_wrench'),
                            ('target_thrust', 'gnc/target_thrust'),
                        ],
                    ),
                    # Launch thruster controller node
                    Node(
                        package='tauv_vehicle',
                        executable='thruster_controller',
                        name='thruster_controller',
                        remappings=[
                            ('target_thrust', 'gnc/target_thrust'),
                            ('thruster_setpoint', 'vehicle/actuators/thruster_setpoint'),
                        ],
                    ),
                    # # Launch wrench test node - publishes test wrench commands
                    # Node(
                    #     package='tauv_common',
                    #     executable='wrench_test',
                    #     name='wrench_test',
                    #     output='screen',
                    # ),
                ]
            ),
        ]
    )
