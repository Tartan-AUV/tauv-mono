# dummy_detector
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
from tauv_msgs.msg import BucketDetection, BucketList, ObjectDetection
from tauv_common.srv import RegisterObjectDetection




class Dummy_Detector():
    def __init__(self):
        rospy.wait_for_service("detector_bucket/register_object_detection")
        self.registration_service = rospy.ServiceProxy("detector_bucket/register_object_detection", RegisterObjectDetection)
        self.registration_test_number = 1

    def spin(self):
        if(self.registration_test_number > 0):
            objdet = ObjectDetection()
            objdet.bucket_detection.tag = "test"
            success = self.registration_service(objdet)
            print("Detection transmitted: " + str(success))





def main():
    rospy.init_node("dummy_detector")
    dummy_detector = Dummy_Detector()
    while not rospy.is_shutdown():
        dummy_detector.spin()
        rospy.sleep(.1)




