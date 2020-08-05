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
from detector_bucket.detector_bucket_utils import *
from scipy.spatial.transform import Rotation as R


imu_data = Imu()
depth_data = 0



class Twist_Transformer:
    def __init__(self):
        self.rotated_twist_pub = rospy.Publisher("rotated_twist", Twist, queue_size=50)
        #self.ekf_sub = rospy.Subscriber()
        self.tf = tf.TransformListener()
        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)


    def spin(self):
        return

def main():
    rospy.init_node('twist_transformer', anonymous=True)
    twist_transformer = Twist_Transformer()
    rospy.spin()

















