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


imu_data = Imu()
depth_data = 0

class Depth_Odom:
    def __init__(self):
        self.depth_data = 0.0
        self.depth_data_variance = .1
        self.depth_data_frame = ""
        print("creating sub")
        self.depth_sensor_sub = rospy.Subscriber("sensors/depth", FluidDepth, self.depth_data_callback)
        print("creating pub")
        self.depth_pose_pub = rospy.Publisher("sensors/depth_pose", PoseWithCovarianceStamped, queue_size=50)
        self.map_broadcaster = tf.TransformBroadcaster()
        rospy.loginfo("finished init")

    def depth_data_callback(self, data):
        self.depth_data = data.depth
        self.depth_data_frame = data.header.frame_id
        self.depth_data_variance = data.variance

    def spin(self):
        pose_rot = quaternion_from_euler(0, 0, 0)
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = self.depth_data_frame
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.pose.position.z = self.depth_data
        pose_msg.pose.covariance[14] = self.depth_data_variance**2
        self.depth_pose_pub.publish(pose_msg)
        self.map_broadcaster.sendTransform((0, 0, 0),
                                           quaternion_from_euler(0, 0, 0),
                                           rospy.Time.now(),
                                           "albatross/pressure_link",
                                           "albatross/odom")

def main():
    rospy.init_node('state_estimation')
    d_odom = Depth_Odom()
    while not rospy.is_shutdown():
        d_odom.spin()
        rospy.sleep(.1)

















