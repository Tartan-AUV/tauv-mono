# mission_planner.py
#
# Yuh.
#
# Author: Neel Gandhi & Rishabh Jain




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

class Task:
  def __init__(self, timeout):
    self.timeout = timeout
    self.status = "Not Started"
    self.done = False
    self.name = ""
    # Declare publishers, subcribers, etc. 

class Mission_Planner:
  def __init__(self, task_list):
    self.task_list = task_list
    self.cur_task = None
    self.pub = rospy.Publisher("mission_output", String, queue_size = 50)

  def run(self):
    for task in self.task_list:
      self.cur_task = task.name
      start_time = rospy.get_time()
      while (rospy.get_time() - start_time < task.timeout) and (not task.done):
        continue
            # do something

  def spin(self):
    self.pub.publish(self.cur_task)



# imu_data = Imu()
# depth_data = 0

# class Depth_Odom:
#     def __init__(self):
#         self.depth_data = 0.0
#         self.depth_data_variance = .1
#         self.depth_data_frame = ""
#         self.imu_data = Imu()
#         self.depth_sensor_sub = rospy.Subscriber("sensors/depth", FluidDepth, self.depth_data_callback)
#         self.imu_sensor_sub = rospy.Subscriber("sensors/imu/data", Imu, self.imu_data_callback)
#         self.depth_odom_pub = rospy.Publisher("sensors/depth_odom", Odometry, queue_size=50)
#         self.map_broadcaster = tf.TransformBroadcaster()

#     def depth_data_callback(self, data):
#         self.depth_data = data.depth
#         self.depth_data_frame = data.header.frame_id
#         self.depth_data_variance = data.variance

#     def imu_data_callback(self, data):
#         self.imu_data = data


#     def spin(self):
#         odom = Odometry()
#         odom.header.stamp = rospy.Time.now()
#         odom.header.frame_id = "odom"
#         odom.child_frame_id = "pressure_link"
#         pose_msg = PoseWithCovarianceStamped()
#         odom.pose.pose.position.z = -self.depth_data
#         odom.pose.covariance[14] = self.depth_data_variance**2
#         self.depth_odom_pub.publish(odom)
#         # q = self.imu_data.orientation
#         # q = [q[0], q[1], q[2], q[3]]
#         # self.map_broadcaster.sendTransform((0, 0, 0),
#         #                                    Quaternion(*q),
#         #                                    rospy.Time.now(),
#         #                                    "albatross/imu_debug",
#         #                                     "albatross/odom")

def main():
    rospy.init_node('mission_planner', anonymous=True)
    d_odom = Depth_Odom()
    while not rospy.is_shutdown():
        d_odom.spin()
        rospy.sleep(.1)

















