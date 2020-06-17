# dummy_detector
#
# This node is for testing the vision bucket registration services
#
#
# Author: Advaith Sethuraman 2020


#!/usr/bin/env python
import rospy
import tf
import tf_conversions
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Imu, Image
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import *
from jsk_recognition_msgs.msg import BoundingBox
from nav_msgs.msg import Odometry
from tf.transformations import *
from std_msgs.msg import *
from geometry_msgs.msg import Quaternion
from tauv_msgs.msg import BucketDetection, BucketList, ObjectDetection
from tauv_common.srv import RegisterObjectDetection


class Dummy_Detector():
    def __init__(self):
        rospy.wait_for_service("detector_bucket/register_object_detection")
        self.registration_service = rospy.ServiceProxy("detector_bucket/register_object_detection", RegisterObjectDetection)
        self.left_stream = rospy.Subscriber("/albatross/stereo_camera_left_under/camera_image", Image, self.left_callback)
        self.right_stream = rospy.Subscriber("/albatross/stereo_camera_right_under/camera_image", Image, self.right_callback)
        self.registration_test_number = 1
        self.stereo_proc = cv2.StereoBM_create(numDisparities=16, blockSize=33)
        self.cv_bridge = CvBridge()
        self.stereo_left = []
        self.left_flag = False

    def left_callback(self, msg):
        self.stereo_left = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        cv2.imwrite("/home/advaith/Desktop/left.png", self.stereo_left)
        self.left_flag = True

    def right_callback(self, msg):
        if(self.left_flag):
            right_img = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
            cv2.imwrite("/home/advaith/Desktop/right.png", right_img)
            disp = self.stereo_proc.compute(cv2.cvtColor(self.stereo_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))
            cv2.normalize(disp, disp, 0, 255, cv2.NORM_MINMAX)
            cv2.imshow("Stereo", disp)
            cv2.waitKey(1)
            self.left_flag = False

    def spin(self):
        if(self.registration_test_number > 0):
            obj_det = ObjectDetection()
            obj_det.bucket_detection.tag = "Testing"
            bbox_3d = BoundingBox()
            bbox_3d.dimensions = Vector3(1, 1, 1)
            bbox_pose = Pose()
            bbox_pose.position.x = 0
            bbox_pose.position.y = 0
            bbox_pose.position.z = 0
            bbox_3d.pose = bbox_pose
            bbox_header = Header()
            bbox_header.frame_id = "odom"
            bbox_3d.header = bbox_header
            obj_det.bucket_detection.bbox_3d = bbox_3d
            success = self.registration_service(obj_det)
            print("Detection transmitted: " + str(success))
            self.registration_test_number -= 1





def main():
    rospy.init_node("dummy_detector")
    dummy_detector = Dummy_Detector()
    while not rospy.is_shutdown():
        dummy_detector.spin()
        rospy.sleep(.1)




