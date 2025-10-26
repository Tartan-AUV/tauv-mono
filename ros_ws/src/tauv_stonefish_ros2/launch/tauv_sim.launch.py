from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(package='tauv_stonefish_ros2', executable='tauv_sim')])
