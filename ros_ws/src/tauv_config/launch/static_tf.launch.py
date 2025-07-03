from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

def generate_launch_description():
    return LaunchDescription([
            # Static transform publisher for DVL frame
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='dvl_static_tf_publisher',
                arguments=[
                    # x, y, z, yaw, pitch, roll, parent_frame, child_frame
                    '-0.103', '0.049', '0.045',  # Translation from scenario file
                    '0.0', '0.0', '1.5708',      # Rotation: 90 degrees around Z-axis (π/2 radians)
                    'os/body', 'os/hull/dvl'
                ],
                output='screen'
            ),
            
            # Static transform publisher for depth sensor frame
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='depth_static_tf_publisher',
                arguments=[
                    # x, y, z, yaw, pitch, roll, parent_frame, child_frame
                    '0.240', '0.113', '-0.055',  # Translation from scenario file
                    '0.0', '0.0', '0.0',         # Rotation (no rotation)
                    'os/body', 'os/hull/depth'
                ],
            output='screen'
        )
    ]) 