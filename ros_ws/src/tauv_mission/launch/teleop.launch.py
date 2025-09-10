from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch the teleop CLI node.

    Running through ROS2 launch allows convenient integration with other launch
    files, while `emulate_tty=True` ensures that the interactive CLI receives a
    proper TTY so that readline features (arrow keys, history) work.
    """
    return LaunchDescription(
        [
            Node(
                package='tauv_mission',
                executable='teleop',
                name='teleop',
                namespace='os',
                output='screen',
                emulate_tty=True,
            ),
        ]
    )
