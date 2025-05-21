from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('stonefish_ros2'),
                    'launch',
                    'stonefish_simulator.launch.py'
                ])
            ]),
            launch_arguments = {
                'simulation_data' : PathJoinSubstitution([FindPackageShare('tauv_sim'), 'data']),
                'scenario_desc' : PathJoinSubstitution([FindPackageShare('tauv_sim'), 'scenarios', 'osprey_irvine.scn']),
                'simulation_rate' : '300.0',
                'window_res_x' : '1920',
                'window_res_y' : '1080',
                'rendering_quality' : 'medium'
            }.items()
        )
    ])