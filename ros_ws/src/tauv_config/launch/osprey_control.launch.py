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

            # Commander node - converts velocity/attitude commands to acceleration commands
            # Node(
            #     package='tauv_common',
            #     executable='commander',
            #     name='commander',
            #     output='screen',
            # ),

            # # # INDI Controller node - converts acceleration commands to wrench commands
            # Node(
            #     package='tauv_common',
            #     executable='indi_controller',
            #     name='indi_controller',
            #     output='screen',
            #     # No remapping needed - topics are already in gnc namespace
            #     parameters=[{
            #         'accel_filter_alpha': 0.5,
            #         'max_force': 100.0,
            #         'max_torque': 50.0
            #     }]
            # ),

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
                package='tauv_vehicle',
                executable='thruster_controller',
                name='thruster_controller',
                remappings=[
                    ('target_thrust', 'gnc/target_thrust'),
                ]
            ),
            
            # Launch wrench test node - publishes test wrench commands
            Node(
                package='tauv_common',
                executable='wrench_test',
                name='wrench_test',
                output='screen',
            ),
        ]),
        
        # Include the static TF publisher launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                str(tauv_config_share / 'launch' / 'static_tf.launch.py')
            ])
        ),
    ])
