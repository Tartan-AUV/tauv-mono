#!/usr/bin/env python3

# detector_bucket
#
# This node is the for aggregating the detections from the vision pipeline.
# Input: Detection
# Output: Daemons publish individual detections to pose_graph
# Author: Advaith Sethuraman 2020


import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from vision.detector_bucket.detector_bucket_utils import *
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from stereo_msgs.msg import DisparityImage
from jsk_recognition_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tauv_msgs.msg import BucketDetection, BucketList, PoseGraphMeasurement,RegisterObjectDetections, RegisterMeasurement
from scipy.spatial.transform import Rotation as R


class Detector_Bucket():
    def __init__(self):
        #rospy.init_node('detector_bucket', anonymous = True)

        self.num_daemons = 1
        self.daemon_names = None
        self.daemon_dict = {}
        if not self.init_daemons():
            rospy.logerr("[Detector Bucket]: Unable to initialize detector daemons, invalid information!")
        rospy.loginfo("[Detector Bucket]: Summoning Daemons: " + str(self.daemon_names))

        self.cv_bridge = CvBridge()

        self.bucket_list = BucketList()

        #rospy.init_node('detector_bucket', anonymous = True)

        self.bucket_list_pub = rospy.Publisher("bucket_list", BucketList, queue_size=50)
        self.detection_server = rospy.Subscriber("register_object_detection", RegisterObjectDetections,
                                              self.update_daemon_service)

    def init_daemons(self):
        if rospy.has_param("detectors/total_number"):
            self.num_daemons = int(rospy.get_param("detectors/total_number"))
            #rospy.loginfo("[Detector Bucket]: Initializing %d Daemons", self.num_daemons)
            self.daemon_names = rospy.get_param("detectors/names")
            #rospy.loginfo(f"{self.daemon_names}")
            self.daemon_dict = {name: Detector_Daemon(name, ii) for ii, name in enumerate(self.daemon_names)}
            print(self.daemon_dict)
            return True
        else:
            return False

    def is_valid_registration(self, new_detection):
        tag = new_detection.tag != ""
        return tag


    def update_daemon_service(self, req):
        data_frame = req.objdets
        
        # acquire and update data buffer on daemon
        daemon_name = req.detector_tag
        #rospy.loginfo(f"dict {self.daemon_dict}\n")
        if daemon_name in self.daemon_dict:
            #rospy.loginfo(f"data = {data_frame}")

            daemon = self.daemon_dict[daemon_name]
            daemon.mutex.acquire()
            daemon.update_detection_buffer(data_frame)
            daemon.mutex.release()

            return True
        return False

    def spin_daemon(self, daemon):
        daemon.mutex.acquire()
        if(daemon.new_data):
            daemon.spin()
        daemon.mutex.release()

    def publish(self, daemon_name = "camera"):
        daemon = self.daemon_dict[daemon_name]
        self.spin_daemon(daemon)
        # rospy.loginfo(f"daemon: {daemon}")
        self.bucket_list_pub.publish(daemon)

    def find(self, daemon_name = "camera"):
        return self.daemon_dict[daemon_name]

    #return the first bucket detection with the key matching name
    def find_by_tag(self, tag, dameon_name = "camera"):
        daemon = self.daemon_dict[daemon_name]
        self.spin_daemon(daemon)
        for entry in daemon.bucket_list:
            if entry.tag == tag:
                return entry
        return None

    #iterate through all daemons and call spin function to update tracking
    def spin_all(self):
        for daemon_name in self.daemon_dict:
            daemon = self.daemon_dict[daemon_name]
            self.spin_daemon(daemon)

    def publish_all(self):
       for daemon_name in self.daemon_dict:
            self.publish(daemon_name)
