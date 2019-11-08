#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import geometry_msgs.msg
import tf_conversions
from tf.transformations import *
from geometry_msgs.msg import Quaternion
import tauv_msgs
import numpy as np
from tauv_msgs.msg import FluidDepth


imu_data = Imu()
depth_data = 0

def imu_data_callback(data):
    global imu_data
    imu_data = data

def depth_data_callback(data):
    global depth_data
    depth_data = data.data
    odom_rot = quaternion_from_euler(0, 0, 0)
    odom = Odometry()
    odom.header.frame_id = data.header.frame_id
    odom.pose.pose = Pose(Point(0.0, 0.0, data.depth), Quaternion(*odom_rot))


        
def main():
    rospy.init_node('state_estimation')
    rospy.Subscriber("sensors/imu_data", Imu, imu_data_callback)
    rospy.Subscriber("sensors/depth_data", FluidDepth, depth_data_callback)
    depth_odom_pub = rospy.Publisher("sensors/depth_odom", Odometry, queue_size=50)
















