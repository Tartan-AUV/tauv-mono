from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for the full Osprey vehicle stack.

    This launch file composes the existing *basic* GNC stack defined in
    `osprey_gnc_debug.launch.py` with the real-time vehicle controller (RTVC)
    node provided by the ``tauv_vehicle`` package.

    Topic remappings ensure that the topics produced/consumed by RTVC line up
    with the names expected by the GNC stack:

    *   IMU           → ``vehicle/sensors/imu``
    *   Temperature   → ``vehicle/sensors/temperature``
    *   Pressure      → ``vehicle/sensors/pressure``
    *   Depth frame   → ``vehicle/sensors/depth``
    *   Thruster cmd  ← ``vehicle/actuators/thruster_setpoint``

    All nodes are launched in the ``os`` namespace to keep the graph tidy.
    """

    # Path to the existing GNC stack launch file
    tauv_config_share = Path(get_package_share_directory("tauv_config"))
    gnc_launch = tauv_config_share / "launch" / "osprey_gnc_debug.launch.py"

    # Include the original GNC stack as-is
    gnc_stack = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gnc_launch)))

    # RTVC node with the necessary topic remappings
    rtvc_node = Node(
        package="tauv_vehicle",
        executable="rtvc",
        name="rtvc",
        namespace="os",
        output="screen",
        remappings=[
            ("imu", "vehicle/sensors/imu"),
            ("temperature", "vehicle/sensors/temperature"),
            ("pressure", "vehicle/sensors/pressure"),
            ("depth", "vehicle/sensors/depth"),
            # RTVC subscribes to this topic – remap so it matches the publisher in
            # the thruster controller.
            ("thruster_setpoint", "vehicle/actuators/thruster_setpoint"),
        ],
    )

    return LaunchDescription(
        [
            gnc_stack,
            rtvc_node,
        ]
    )
