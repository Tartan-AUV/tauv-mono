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
import matplotlib.pyplot as plt
import matplotlib.animation


imu_data = Imu()
depth_data = 0

class Visualizer:
    def __init__(self):
        self.depth_pose = Pose()
        self.x = 3.0
        self.y = 0.0
        print("creating sub")
        self.depth_pose_sub = rospy.Subscriber("albatross/sensors/depth_pose", PoseStamped, self.depth_pose_callback)
        print("creating pub")
        self.fig, (self.ax1, self.ax2) = plt.subplots(nrows=2)
        self.line1, = self.ax1.plot(self.x, self.y)
        self.ani = matplotlib.animation.FuncAnimation(self.fig, self.update)
        self.width = 10
        plt.show()

    def depth_pose_callback(self, data):
        self.depth_pose = data

    def update(self, i):
        self.y = [self.depth_pose.pose.position.z]*(float(self.width)/.2)
        self.x = np.arange(0., float(self.width), .2)

        self.line1.set_data(self.x, self.y)



def main():
    rospy.init_node('state_estimation_visualizer')
    print("Created Visualizer")
    d_odom = Visualizer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'ThrusterAllocatorNode::Exception'

















