# detector_bucket
#
# This node is the for aggregating the detections from the vision pipeline.
# The information in the bucket will be broadcast to the mission nodes and used for tasks.
# Main outputs are a 3D Bounding Box array and a list of bucket detections for mission nodes
# New detections will be added to observations using a pose graph.
#
# Author: Advaith Sethuraman 2020


#!/usr/bin/env python
import rospy
import tf
import tf_conversions
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from stereo_msgs.msg import DisparityImage
from jsk_recognition_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import *
from nav_msgs.msg import Odometry
from tf.transformations import *
from geometry_msgs.msg import Quaternion
from tauv_msgs.msg import BucketDetection, BucketList
from tauv_common.srv import RegisterObjectDetection





class Detector_Bucket():
    def __init__(self):
        self.bucket_list_pub = rospy.Publisher("bucket_list", BucketList, queue_size=50)
        self.bbox_3d_list_pub = rospy.Publisher("bucket_bbox_3d_list", BoundingBoxArray, queue_size=50)
        self.detection_server = rospy.Service("detector_bucket/register_object_detection", RegisterObjectDetection, self.register_object_detection)
        self.tf = tf.TransformListener()
        self.cv_bridge = CvBridge()
        self.refresh_rate = 0
        self.bucket_list = []
        self.bbox_3d_list = []

    def similarity_index(self, detection_1, detection_2):
        point_1 = np.asarray([detection_1.position.x, detection_1.position.y, detection_1.position.z])
        point_2 = np.asarray([detection_2.position.x, detection_2.position.y, detection_2.position.z])
        orb = cv2.ORB_create(300)
        image_1 = self.cv_bridge.imgmsg_to_cv2(detection_1.image, "passthrough")
        image_2 = self.cv_bridge.imgmsg_to_cv2(detection_2.image, "passthrough")
        k1, d1 = orb.detectAndCompute(image_1, None)
        k2, d2 = orb.detectAndCompute(image_2, None)
        matcher = cv2.BFMatcher()
        matches = matcher.match(d1,d2)

        distance = (point_1 - point_2).dot((point_1 - point_2).T)

        return distance

    def is_valid_registration(self, new_detection):
        return True
        for detections in self.bucket_list:
            sim_index = self.similarity_index(detections, new_detection)
            if(sim_index < 1.0):
                print("Similarity Found")
                #send updated landmark registration to pose_graph
                return False
        #send new landmark to pose_graph
        print("New Landmark")
        return True

    def register_object_detection(self, req):
        bucket_detection = req.objdet
        bbox_3d_detection = bucket_detection.bbox_3d
        if(self.is_valid_registration(bucket_detection)):
            found_in_current = False
            for det, bbox in zip(self.bucket_list, self.bbox_3d_list):
                if(bucket_detection.tag == det.tag):
                    det = bucket_detection
                    bbox = bbox_3d_detection
                    found_in_current = True
            if(not found_in_current):
                self.bucket_list.append(bucket_detection)
                self.bbox_3d_list.append(bbox_3d_detection)
            return True
        return False

    def spin(self):
        bucket_list_msg = BucketList()
        bbox_3d_list_msg = BoundingBoxArray()
        bucket_list_msg.header = Header()
        bucket_list_msg.bucket_list = self.bucket_list
        bbox_3d_list_msg.header = Header()
        bbox_3d_list_msg.header.frame_id = "odom"
        bbox_3d_list_msg.boxes = self.bbox_3d_list
        self.bucket_list_pub.publish(bucket_list_msg)
        self.bbox_3d_list_pub.publish(bbox_3d_list_msg)
        return

def main():
    rospy.init_node('detector_bucket', anonymous=True)
    detector_bucket = Detector_Bucket()
    while not rospy.is_shutdown():
        detector_bucket.spin()
        rospy.sleep(.1)



