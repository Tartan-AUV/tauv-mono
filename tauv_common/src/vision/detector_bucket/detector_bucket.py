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
        self.detection_server = rospy.Service("detector_bucket/register_object_detection", RegisterObjectDetection, \
                                              self.register_object_detection)
        self.tf = tf.TransformListener()
        self.cv_bridge = CvBridge()
        self.refresh_rate = 0
        self.bucket_dict = dict()
        self.bbox_3d_list = []
        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)
        self.monotonic_det_id = -1
        self.nn_threshold = .5

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
            if(sim_index < 1.0): #new landmark
                #send updated landmark registration to pose_graph
                return False
        return True

    # returns nearest landmark neighbor in bucket or -1 to signify a new marker
    def find_nearest_neighbor(self, bucket_detection):
        if len(self.bucket_dict.keys()) > 0:
            curr_det_positions = [self.bucket_dict[id][0].position for id in self.bucket_dict]
            curr_det_positions = np.asarray(list(map(lambda pos: np.asarray([pos.x, pos.y, pos.z]), curr_det_positions))).T
            new_det_position = np.asarray([bucket_detection.position.x, bucket_detection.position.y, bucket_detection.position.z]).T
            print("Current loc: " + str(new_det_position))
            print("All others: " + str(curr_det_positions))
            diff = np.asmatrix(new_det_position[:, None] - curr_det_positions)
            print("diff:" + str(diff))
            mahalanobis_distance = np.sqrt(np.diag(diff.T*np.eye(3)*diff))
            nearest_neighbor = np.argmin(mahalanobis_distance)
            # print(" Decidision: %d" % nearest_neighbor)
            print(mahalanobis_distance.size)
            print("maha:" + str(mahalanobis_distance))
            if mahalanobis_distance[nearest_neighbor] < self.nn_threshold and False:
                return nearest_neighbor
        self.monotonic_det_id += 1
        return self.monotonic_det_id

    def register_object_detection(self, req):
        bucket_detection = req.objdet
        bbox_3d_detection = bucket_detection.bbox_3d
        det_id = self.find_nearest_neighbor(bucket_detection)
        print("Decisino: %d", det_id)
        self.bucket_dict[det_id] = (bucket_detection, bbox_3d_detection)
        return True

    def spin(self, event):
        bucket_list_msg = BucketList()
        bbox_3d_list_msg = BoundingBoxArray()
        bucket_list_msg.header = Header()
        bucket_list_msg.bucket_list = [self.bucket_dict[id][0] for id in self.bucket_dict]
        bbox_3d_list_msg.header = Header()
        bbox_3d_list_msg.header.frame_id = "odom"
        bbox_3d_list_msg.boxes = [self.bucket_dict[id][1] for id in self.bucket_dict]
        self.bucket_list_pub.publish(bucket_list_msg)
        self.bbox_3d_list_pub.publish(bbox_3d_list_msg)
        return

def main():
    rospy.init_node('detector_bucket', anonymous=True)
    detector_bucket = Detector_Bucket()
    rospy.Timer(rospy.Duration(.1), detector_bucket.spin)
    rospy.spin()




