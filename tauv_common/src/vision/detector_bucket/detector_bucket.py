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

class Pose_Graph_Node():
    def __init__(self):

    def add_features(self):
        return

    def get_features(self):
        return

    def get_similarity(self, pg_node):
        return

    def get_transform(self, pg_node):
        return

    def set_pose_from_transform(self, parent_pg_node, transform):
        return

class Kd_Tree():
    def __init__(self):

    def add_point(self):
        return

    def get_neighbors(self, radius):
        return

    def recompute_boundaries(self):
        return

class Pose_Graph():
    def __init__(self):
        self.front_image_feed = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_image", Image, self.left_callback)

    def add_pose(self):
        #adds a new pose after a given space/time
        return

    def exists_near_neighbor(self):
        #finds the closest spatial pose graph node
        return

    def compute_loop_closure(self):
        #computes new transform between two nodes
        return

    def compute_node_similarity(self):
        #for vision based BoW or other feature matching technique
        return

    def remove_pose(self):
        #for pruning the pose graph
        return

class Detector_Bucket():
    def __init__(self):
        self.bucket_list_pub = rospy.Publisher("bucket_list", BucketList, queue_size=50)
        self.bbox_3d_list_pub = rospy.Publisher("bucket_bbox_3d_list", BoundingBoxArray, queue_size=50)
        self.detection_server = rospy.Service("detector_bucket/register_object_detection", RegisterObjectDetection, self.register_object_detection)
        self.refresh_rate = 0
        self.bucket_list = []
        self.bbox_3d_list = []

    def is_valid_registration(self, bucket_detection):
        #TODO add system to determine if this is a new detection
        return True

    def register_object_detection(self, req):
        objdet = req.objdet
        bucket_detection = objdet.bucket_detection
        bbox_3d_detection = bucket_detection.bbox_3d
        if(self.is_valid_registration(objdet)):
            self.bucket_list.append(bucket_detection)
            self.bbox_3d_list = []
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



