#!/usr/bin/env python3

import threading

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from tauv_msgs.msg import Depth, DepthSensorFrame


class DepthEstimatorNode(Node):
    def __init__(self):
        super().__init__('depth_estimator')

        # Declare parameters with default values
        self.declare_parameter('surface_pressure', 101325.0)
        self.declare_parameter('water_density', 997.0)
        self.declare_parameter('gravity', 9.81)
        self.declare_parameter('variance', 1.0e-4)

        self.surface_pressure = (
            self.get_parameter('surface_pressure').get_parameter_value().double_value
        )
        self.water_density = self.get_parameter('water_density').get_parameter_value().double_value
        self.gravity = self.get_parameter('gravity').get_parameter_value().double_value
        self.variance = self.get_parameter('variance').get_parameter_value().double_value

        self._surface_pressure_lock = threading.Lock()
        self._reset_triggered = False
        self._reset_lock = threading.Lock()

        self.depth_pub = self.create_publisher(Depth, 'depth', 10)
        self.depth_sub = self.create_subscription(
            DepthSensorFrame, 'depth_sensor_frame', self.depth_sensor_frame_callback, 10
        )
        self.reset_srv = self.create_service(Trigger, 'reset_depth', self.handle_reset_service)

    def depth_sensor_frame_callback(self, msg: DepthSensorFrame):
        # with self._reset_lock:
        #     if self._reset_triggered:
        #         with self._surface_pressure_lock:
        #             self.surface_pressure = float(msg.pressure)
        #         self._reset_triggered = False

        # with self._surface_pressure_lock:
        #     surface_pressure = self.surface_pressure

        depth_msg = Depth()
        depth_msg.header = msg.header
        # depth_msg.depth = (float(msg.pressure) - surface_pressure) / (self.water_density * self.gravity)
        depth_msg.depth = float(msg.depth)
        depth_msg.variance = self.variance

        self.depth_pub.publish(depth_msg)

    def handle_reset_service(self, request, response):
        with self._reset_lock:
            self._reset_triggered = True
        response.success = True
        response.message = 'Reset triggered'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
