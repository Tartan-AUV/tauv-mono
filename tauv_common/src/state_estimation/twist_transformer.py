# twist_transformer
#
# Transforms twists in base_link to a global frame
#
#
# Author: Advaith Sethuraman 2020


#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import *
from nav_msgs.msg import Odometry
import tf_conversions
from tf.transformations import *
import tf
from geometry_msgs.msg import Quaternion
import tauv_msgs
import numpy as np
from tauv_msgs.msg import FluidDepth
from tauv_util.types import tl, tm
from vision.detector_bucket.detector_bucket_utils import *
from scipy.spatial.transform import Rotation as R

class Twist_Transformer:
    def __init__(self):
        self.rotated_twist_pub = rospy.Publisher("odometry/filtered/rotated", Odometry, queue_size=50)
        self.ekf_sub = rospy.Subscriber("odometry/filtered", Odometry, self.odom_callback)
        self.tf = tf.TransformListener()
        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)

    def transform_meas_to_world(self, measurement, child_frame, world_frame, time, translate=True):
        self.tf.waitForTransform(world_frame, child_frame, time, rospy.Duration(4.0))
        try:
            (trans, rot) = self.tf.lookupTransform(world_frame, child_frame, time)
            tf = R.from_quat(np.asarray(rot))
            detection_pos = tf.apply(measurement)
            if translate:
                detection_pos += np.asarray(trans)
            return detection_pos
        except:
            return np.array([np.nan])

    def odom_callback(self, msg):
        odom_rotated = Odometry()
        odom = msg
        pose = odom.pose
        twist = odom.twist

        #tranform twist into odom frame
        linear_twist = twist.twist.linear
        angular_twist = twist.twist.angular

        twist_rotated = TwistWithCovariance()
        transformed_linear = self.transform_meas_to_world(tl(linear_twist), odom.child_frame_id, odom.header.frame_id, rospy.Time(0), False)
        transformed_angular = self.transform_meas_to_world(tl(angular_twist), odom.child_frame_id, odom.header.frame_id, rospy.Time(0), False)
        twist_rotated.twist.linear = tm(transformed_linear, Vector3)
        twist_rotated.twist.angular = tm(transformed_angular, Vector3)
        twist_rotated.covariance = twist.covariance #may need to be transformed

        odom_rotated.header = odom.header
        odom_rotated.child_frame_id = odom.child_frame_id
        odom_rotated.pose = pose
        odom_rotated.twist = twist_rotated

        self.rotated_twist_pub.publish(odom_rotated)

    def spin(self):
        return

def main():
    rospy.init_node('twist_transformer', anonymous=True)
    twist_transformer = Twist_Transformer()
    rospy.spin()

















