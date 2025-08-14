from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace, Node


def generate_launch_description():
    return LaunchDescription([
        GroupAction(actions=[
            PushRosNamespace("os"),

            # Commander node - converts velocity/attitude commands to acceleration commands
            # Node(
            #     package='tauv_common',
            #     executable='commander',
            #     name='commander',
            #     output='screen',
            #     # No remapping needed - topics are already in gnc namespace
            #     parameters=[{
            #         'kp_velocity': 2.0,
            #         'kd_velocity': 0.1,
            #         'kp_attitude': 1.5,
            #         'kd_attitude': 0.3,
            #         'max_linear_accel': 2.0,
            #         'max_angular_accel': 1.0
            #     }]
            # ),

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
    ])
