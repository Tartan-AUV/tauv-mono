#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Twist, Vector3
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
        self.depth_data = FluidDepth()
        self.depth_data_frame = ""
        print("creating sub")
        self.depth_sensor_sub = rospy.Subscriber("sensors/depth", FluidDepth, self.depth_data_callback)
        print("creating pub")
        self.depth_pose_pub = rospy.Publisher("sensors/depth_pose", PoseStamped, queue_size=50)
        self.map_broadcaster = tf.TransformBroadcaster()
        rospy.loginfo("finished init")

    def depth_data_callback(self, data):
        self.depth_data = data.depth
        self.depth_data_frame = data.header.frame_id

    def spin(self):
        pose_rot = quaternion_from_euler(0, 0, 0)
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.depth_data_frame
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose = Pose(Point(0.0, 0.0, self.depth_data), Quaternion(*pose_rot)) #figure out why the depth data is throwing an error
        self.depth_pose_pub.publish(pose_msg)
        self.map_broadcaster.sendTransform((1, 0, 0),
                                           quaternion_from_euler(0, 0, 0),
                                           rospy.Time.now(),
                                           "albatross/odom",
                                           "/map")

def main():
    rospy.init_node('state_estimation')
    d_odom = Depth_Odom()
    while not rospy.is_shutdown():
        d_odom.spin()
        rospy.sleep(.1)

















