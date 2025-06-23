from typing import Optional

import numpy as np
import scipy as sp
import spatialmath.quaternion
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from spatialmath import SE3, SO3
import tf2_ros as tf2

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tauv_msgs.msg import Depth, WaterlinkedDvlFrame

from iekf.system import IEKFSystemKinematic
from iekf.iekf import IEKF

def transform_to_SE3(t):
    q, t = t.rotation, t.translation
    t = np.array([t.x, t.y, t.z])
    R = spatialmath.quaternion.UnitQuaternion(q.w, (q.x, q.y, q.z)).SO3()
    return SE3.Rt(R, t)


class StateEstimatorEkf(Node):
    def __init__(self):
        super().__init__('state_estimator_ekf')

        # Setup publishers and subscribers
        # TODO: are these QoS profiles good?
        self.create_subscription(Imu, "sensors/imu", self._imu_msg_callback, qos_profile=10)
        self.create_subscription(WaterlinkedDvlFrame, "sensors/dvl", self._dvl_msg_callback, qos_profile=10)
        self.create_subscription(Depth, "sensors/depth", self._depth_msg_callback, qos_profile=10)
        self.create_publisher(Odometry, "odom", qos_profile=10)

        # Transforms
        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer, self)
        self._tf_broadcaster: tf2.TransformBroadcaster = tf2.TransformBroadcaster(self)

        # Parameters
        self._body_frame = (
            self.declare_parameter("body_frame").get_parameter_value().string_value
        )
        self._dvl_frame = (
            self.declare_parameter("dvl_frame").get_parameter_value().string_value
        )
        self._depth_frame = (
            self.declare_parameter("depth_frame").get_parameter_value().string_value
        )
        self._initial_position_stddev = (
            self.declare_parameter("initial_position_stddev_m", 0.01).get_parameter_value().double_value
        )
        self._initial_orientation_stddev = np.deg2rad(
            self.declare_parameter("initial_orientation_stddev_deg", 20.0).get_parameter_value().double_value
        )
        self._initial_linear_velocity_stddev = (
            self.declare_parameter("initial_linear_velocity_stddev_mps", 0.3).get_parameter_value().double_value
        )
        # TODO: Add units on all these values. Make defaults sensible. Replace variance with stddev.
        self._gyro_bias_rw_cov = (
            np.eye(3) *
            self.declare_parameter("gyro_bias_rw_var", 1e-12).get_parameter_value().double_value
        )
        self._accel_bias_rw_cov = (
            np.eye(3) *
            self.declare_parameter("accel_bias_rw_var", 1e-12).get_parameter_value().double_value
        )
        self._position_process_noise_cov = (
            np.eye(3) *
            self.declare_parameter("position_process_noise_stddev", 1e-6).get_parameter_value().double_value  ** 2
        )
        self._velocity_process_noise_cov = (
            np.eye(3) *
            self.declare_parameter("velocity_process_noise_stddev", 1e-6).get_parameter_value().double_value ** 2
        )
        self._orientation_process_noise_cov = (
            np.eye(3) *
            self.declare_parameter("orientation_process_noise_stddev", 1e-6).get_parameter_value().double_value ** 2
        )

        # Static transforms
        self._body_T_dvl: Optional[SE3] = None
        self._r_body_depth_B: Optional[SE3] = None
        self._static_tf_lookup_timer = self.create_timer(0.1, self._get_static_transforms)

        # Filter
        self._system: Optional[IEKFSystemKinematic] = None
        self._iekf: Optional[IEKF] = None


    def _get_static_transforms(self):
        try:
            now = rclpy.time.Time()
            body_T_dvl = self._tf_buffer.lookup_transform(
                self._body_frame,
                self._dvl_frame,
                now,
            ).transform
            self._body_T_dvl = transform_to_SE3(body_T_dvl)

            t = self._tf_buffer.lookup_transform(
                self._body_frame,
                self._depth_frame,
                now,
            ).transform.translation
            self._r_body_depth_B = np.array([t.x, t.y, t.z])

            self._static_tf_lookup_timer.reset()

            self._init_iekf()

        except tf2.TransformException as ex:
            self.get_logger().debug(f"Error getting static transforms, retrying: {ex}")


    def _imu_msg_callback(self, msg: Imu):
        pass

    def _dvl_msg_callback(self, msg: WaterlinkedDvlFrame):
        pass


    def _depth_msg_callback(self, msg: Depth):
        pass

    def _init_iekf(self):
        # Construct process noise matrix
        Q = sp.linalg.block_diag(
            self._initial_orientation_stddev
        )

        # TODO: We should be using measurement covariance from messages
        self._system = IEKFSystemKinematic()
        pass


if __name__ == "__main__":
    rclpy.init()
    node = StateEstimatorEkf()
    rclpy.spin(node)
    rclpy.shutdown()
