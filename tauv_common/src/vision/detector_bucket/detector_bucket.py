# detector_bucket
#
# This node is the for aggregating the detections from the vision pipeline.
# The information in the bucket will be broadcast to the mission nodes and used for tasks.
#
# Author: Advaith Sethuraman 2020


#!/usr/bin/env python
import rospy
import tf
import tf_conversions
import numpy as np
from sensor_msgs.msg import Imu
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import *
from nav_msgs.msg import Odometry
from tf.transformations import *
from geometry_msgs.msg import Quaternion
from tauv_msgs.msg import BucketDetection, BucketList
from tauv_common.srv import RegisterObjectDetection


class Detector_Bucket():
    def __init__(self):
        self.bucket_list_pub = rospy.Publisher("bucket_list", BucketList, queue_size=50)
        self.detection_server = rospy.Service("detector_bucket/register_object_detection", RegisterObjectDetection, self.register_object_detection)
        self.disp = rospy.Subscriber("disparity", DisparityImage, self.callback)
        self.refresh_rate = 0
        self.detections = []

    def callback(self, img):
        return

    def is_valid_registration(self, bucket_detection):
        return True

    def register_object_detection(self, req):
        bucket_detection = req.bucket_detection
        img = req.image
        bbox_2d = req.bbox_2d
        if(self.is_valid_registration(bucket_detection)):
            self.detections.append(bucket_detection)
            return True
        return False

    def spin(self):
        self.bucket_list_pub.publish(self.detections)
        return

def main():
    rospy.init_node('detector_bucket', anonymous=True)
    detector_bucket = Detector_Bucket()
    while not rospy.is_shutdown():
        detector_bucket.spin()
        rospy.sleep(.1)



