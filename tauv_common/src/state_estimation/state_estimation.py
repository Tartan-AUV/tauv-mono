#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
import tf_conversions
from tf.transformations import *
from geometry_msgs.msg import Quaternion
import tauv_msgs
import numpy as np
from tauv_msgs.msg import FluidDepth


imu_data = Imu()
depth_data = 0

class Depth_Odom:
    def __init__(self):
        self.depth_data = 0.0
        rospy.Subscriber("sensors/depth", FluidDepth, self.depth_data_callback)
        self.depth_odom_pub = rospy.Publisher("sensors/depth_odom", Odometry, queue_size=50)

    def depth_data_callback(self, data):
        self.depth_data = data

    def spin(self):
        odom_rot = quaternion_from_euler(0, 0, 0)
        odom = Odometry()
        odom.header.frame_id = "pressure_link"
        odom.pose.pose = Pose(Point(0.0, 0.0, self.depth_data.depth), Quaternion(*odom_rot)) #figure out why the depth data is throwing an error
        odom.child_frame_id = "base_link"
        odom.twist.twist = Twist(Vector3(0,0,0), Vector3(0, 0, 0))
        self.depth_odom_pub.publish(odom)

def main():
    rospy.init_node('state_estimation')
    d_odom = Depth_Odom()
    while not rospy.is_shutdown():
        d_odom.spin()


















