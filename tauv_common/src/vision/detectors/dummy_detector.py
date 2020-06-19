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
from sensor_msgs.msg import Imu, Image, CameraInfo
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
        self.left_stream = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_image", Image, self.left_callback)
        self.disparity_stream = rospy.Subscriber("/vision/front/disparity", DisparityImage, self.disparity_callback)
        self.left_camera_info = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_info", CameraInfo, self.camera_info_callback)
        #self.right_stream = rospy.Subscriber("/albatross/stereo_camera_right_under/camera_image", Image, self.right_callback)
        self.registration_test_number = 1
        self.stereo_proc = cv2.StereoBM_create(numDisparities=16, blockSize=33)
        self.cv_bridge = CvBridge()
        self.orb = cv2.ORB_create(1000)
        self.stereo_left = Image()
        self.stereo_right = []
        self.disparity = DisparityImage()
        self.camera_info = CameraInfo()
        self.left_img_flag = False

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def disparity_callback(self, msg):
        self.disparity = msg

    def left_callback(self, msg):
        self.stereo_left = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.left_img_flag = True

    def get_feature_centroid(self, depth_map, keypoints):
        if(len(keypoints) > 0):
            pixel_locations = np.asarray([kp.pt for kp in keypoints]).astype(int)
            depths = self.query_depth_map(depth_map, pixel_locations)
            camera_instrinsic_matrix = np.asarray(self.camera_info.K).reshape((3,3)).T
            pixel_homogeneous = np.concatenate((pixel_locations, np.ones((pixel_locations.shape[0], 1)).astype(int)),axis=1)
            world_locations = camera_instrinsic_matrix.dot(pixel_homogeneous.T).T
            raw_3d_points = np.multiply(np.expand_dims(depths, axis=1), (world_locations / np.expand_dims(np.linalg.norm(world_locations, axis=1), axis=1)))
            raw_3d_points = raw_3d_points[~np.any(np.isinf(raw_3d_points), axis=1), :]
            print(raw_3d_points.shape)
            return np.mean(raw_3d_points, axis=0)
        else:
            return np.asarray([0, 0, 0])

    def query_depth_map(self, map, pixel):

        return map[pixel[:,1], pixel[:,0]]

    def depth_from_disparity(self, msg):
        focal_length = msg.f
        baseline = msg.T
        disparity_image = self.cv_bridge.imgmsg_to_cv2(msg.image, "passthrough")
        return focal_length * baseline / disparity_image

    def spin(self):
        if(self.left_img_flag):
            keypoints, descriptors = self.orb.detectAndCompute(self.stereo_left, None)
            feature_centroid = self.get_feature_centroid(self.depth_from_disparity(self.disparity), keypoints)
            print(feature_centroid)
            cv2.imshow("Features", cv2.drawKeypoints(self.stereo_left, keypoints, None))
            cv2.waitKey(1)
            self.left_img_flag = False
            obj_det = ObjectDetection()
            obj_det.bucket_detection.tag = "Testing"
            bbox_3d = BoundingBox()
            bbox_3d.dimensions = Vector3(1, 1, 1)
            bbox_pose = Pose()
            x, y, z = feature_centroid
            bbox_pose.position = Vector3(x, y, z)
            bbox_3d.pose = bbox_pose
            bbox_header = Header()
            bbox_header.frame_id = "base_link"
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




