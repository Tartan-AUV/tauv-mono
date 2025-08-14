"""
Launch file for the Lucid camera driver C++ node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for lucid_driver_cpp node."""
    
    # Declare launch arguments
    camera_ip_arg = DeclareLaunchArgument(
        'camera_ip',
        default_value='10.0.2.11',
        description='IP address of the Lucid camera'
    )
    
    topic_name_arg = DeclareLaunchArgument(
        'topic_name',
        default_value='/image_raw',
        description='Topic name for publishing images'
    )
    
    horizontal_binning_arg = DeclareLaunchArgument(
        'horizontal_binning',
        default_value='2',
        description='Horizontal binning factor'
    )
    
    vertical_binning_arg = DeclareLaunchArgument(
        'vertical_binning',
        default_value='2',
        description='Vertical binning factor'
    )
    
    vpi_backend_arg = DeclareLaunchArgument(
        'vpi_backend',
        default_value='cuda',
        description='VPI backend to use (cuda, vic, or cpu)'
    )
    
    # Create the node
    lucid_driver_node = Node(
        package='tauv_vehicle',
        executable='lucid_driver_cpp',
        name='lucid_driver_cpp',
        output='screen',
        parameters=[{
            'camera_ip': LaunchConfiguration('camera_ip'),
            'topic_name': LaunchConfiguration('topic_name'),
            'horizontal_binning': LaunchConfiguration('horizontal_binning'),
            'vertical_binning': LaunchConfiguration('vertical_binning'),
            'vpi_backend': LaunchConfiguration('vpi_backend'),
        }]
    )
    
    return LaunchDescription([
        camera_ip_arg,
        topic_name_arg,
        horizontal_binning_arg,
        vertical_binning_arg,
        vpi_backend_arg,
        lucid_driver_node
    ])