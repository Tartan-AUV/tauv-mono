# detector_bucket
#
# This node is the for aggregating the detections from the vision pipeline.
# The information in the bucket will be broadcast to the mission nodes and used for tasks.
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
from tauv_msgs.msg import BucketDetection, BucketList
import numpy as np



class Detector_Bucket():
    def __init__(self):
        self.depth_odom_pub = rospy.Publisher("vision/bucket_list", BucketList, queue_size=50)
        self.refresh_rate = 0 #set this using params in future

    def spin(self):
        #use frame_id in the header
        d = BucketDetection()
        d.info = "testing"
        d2 = BucketDetection()
        d2.info = "lol"
        self.depth_odom_pub.publish([d, d2])
        return

def main():
    rospy.init_node('detector_bucket', anonymous=True)
    detector_bucket = Detector_Bucket()
    while not rospy.is_shutdown():
        detector_bucket.spin()
        rospy.sleep(.1)



