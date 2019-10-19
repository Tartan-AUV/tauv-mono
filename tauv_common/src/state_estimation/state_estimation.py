#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import geometry_msgs.msg
import tf_conversions
import tf2_ros

imu_data = Imu()
depth_data = 0

def imu_data_callback(data):
    global imu_data
    imu_data = data

def depth_data_callback(data):
    global depth_data
    depth_data = data.data

def main():
    rospy.init_node("state_estimation")
    rospy.Subscriber("sensors/imu_data", imu_data_callback)
    rospy.Subscriber("sensors/depth_data", depth_data_callback)
    tf_broadcast = tf2_ros.TransformBroadcaster()
    while not rospy.is_shutdown():
        ###Prediction Step
        
        ###Update (Correction Step)



        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"
        t.transform.translation.z = -depth_data










