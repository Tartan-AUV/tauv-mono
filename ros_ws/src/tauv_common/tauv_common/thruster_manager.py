import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
import numpy as np
from typing import Optional, List
import tf2_ros as tf2
from tauv_common.util.geometry import numpify
from tauv_common.util.time import Duration
from spatialmath import SE3
from geometry_msgs.msg import WrenchStamped
from tauv_msgs.msg import TargetThrust
from std_msgs.msg import Float64

THRUSTER_IDS = ["flh", "flv", "alv", "alh", "arh", "arv", "frv", "frh"]
TF_NAMESPACE = "os"

class ThrusterManager(Node):

    def __init__(self):
        super().__init__("thruster_manager")

        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer, self)

        self._wrench_sub = self.create_subscription(WrenchStamped, 'target_wrench', self._handle_wrench, 10)
        self._target_thrust_pub = self.create_publisher(TargetThrust, 'target_thrust', 10)
        
        self._thruster_frames = [f"{TF_NAMESPACE}/thruster/{thruster_id}" for thruster_id in THRUSTER_IDS]
        self._n_thrusters = len(THRUSTER_IDS)

    def _handle_wrench(self, wrench_stamped: WrenchStamped):
        # Get target wrench
        V_B = numpify(wrench_stamped.wrench)
        wrench_frame = wrench_stamped.header.frame_id
        
        # Get thruster transforms
        W_T_T = SE3.Alloc(self._n_thrusters)
        for (i, thruster_frame) in enumerate(self._thruster_frames):
            try:
                transform = self._tf_buffer.lookup_transform(
                    wrench_frame,
                    thruster_frame, 
                    self.get_clock().now(), 
                    Duration(seconds=0)
                )
                W_T_T[i] = numpify(transform.transform)
            except Exception as e:
                self.get_logger().error(f"Could not get transform from {wrench_frame} to {thruster_frame}: {e}")
                return
        
        # Thruster assignment map
        # If F: (n_thrusters, 1) is the thrust vector, then
        # V_B = M @ F
        M = np.zeros((6, self._n_thrusters)) 
        f_unit_T = np.array([1, 0, 0])

        for (i, thruster_frame) in enumerate(self._thruster_frames):
            f_i_W = W_T_T.R[i] @ f_unit_T
            r_wt_T = W_T_T.t[i]
            tau_i_W = np.cross(r_wt_T, f_i_W)
            M[:, i] = np.hstack((f_i_W, tau_i_W))
        
        # Solve for thrusts
        F = np.linalg.pinv(M) @ V_B
        
        # Publish target thrusts
        target_thrust = TargetThrust()
        target_thrust.target_thrust = F.flatten().tolist()
        self._target_thrust_pub.publish(target_thrust)


def main():
    rclpy.init()
    node = ThrusterManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()