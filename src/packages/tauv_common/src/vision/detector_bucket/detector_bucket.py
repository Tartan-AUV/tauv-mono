# detector_bucket
#
# This node is the for aggregating the detections from the vision pipeline.
# Input: Detection
# Output: Daemons publish individual detections to pose_graph
# Author: Advaith Sethuraman 2020


#!/usr/bin/env python
import rospy
#import tf
#import tf_conversions
import numpy as np
import cv2
from cv_bridge import CvBridge
from detector_bucket_utils import *
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from stereo_msgs.msg import DisparityImage
from jsk_recognition_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import *
from nav_msgs.msg import Odometry
from tf.transformations import *
from geometry_msgs.msg import Quaternion
from tauv_msgs.msg import BucketDetection, BucketList, PoseGraphMeasurement
from tauv_msgs.srv import RegisterObjectDetections, RegisterMeasurement
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R


class Detector_Bucket():
    def __init__(self):
        self.num_daemons = 1
        self.daemon_names = None
        self.daemon_dict = {}
        if not self.init_daemons():
            rospy.logerr("[Detector Bucket]: Unable to initialize detector daemons, invalid information!")
        rospy.loginfo("[Detector Bucket]: Summoning Daemons: " + str(self.daemon_names))

        self.tf = tf.TransformListener()
        self.cv_bridge = CvBridge()
        self.refresh_rate = 0
        self.bucket_dict = dict()
        self.bbox_3d_list = []
        self.monotonic_det_id = -1
        self.nn_threshold = .9
        self.debouncing_threshold = 10
        self.arrow_dict = {}
        self.debouncing_tracker_dict = {}
        self.debounced_detection_dict = {}
        self.total_number_detection_dict = {}

        self.bucket_list_pub = rospy.Publisher("bucket_list", BucketList, queue_size=50)
        self.bbox_3d_list_pub = rospy.Publisher("bucket_bbox_3d_list", BoundingBoxArray, queue_size=50)
        self.detection_server = rospy.Subscriber("register_object_detection", RegisterObjectDetections, \
                                              self.update_daemon_service)
        self.arrow_pub = rospy.Publisher("detection_marker", MarkerArray, queue_size=10)
        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)

    def init_daemons(self):
        if rospy.has_param("detectors/total_number"):
            self.num_daemons = int(rospy.get_param("detectors/total_number"))
            rospy.loginfo("[Detector Bucket]: Initializing %d Daemons", self.num_daemons)
            self.daemon_names = rospy.get_param("detectors/names")
            self.daemon_dict = {name: Detector_Daemon(name, ii) for ii, name in enumerate(self.daemon_names)}
            return True
        else:
            return False

    def is_valid_registration(self, new_detection):
        tag = new_detection.tag != ""
        return tag

    def update_detection_arrows(self, bucket_detection, world_frame, robot_position, id):
        pos = bucket_detection.position
        m = Marker()
        m.header.frame_id = world_frame
        m.id = id
        m.points = [robot_position, pos]
        m.color.g = 1.0
        m.color.a = 1.0
        m.scale.x = .05
        m.scale.y = .05
        m.scale.z = .05
        self.arrow_dict[id] = m

    """def transform_meas_to_world(self, measurement, child_frame, world_frame, time, translate=True):
        self.tf.waitForTransform(world_frame, child_frame, time, rospy.Duration(4.0))
        try:
            (trans, rot) = self.tf.lookupTransform(world_frame, child_frame, time)
            tf = R.from_quat(np.asarray(rot))
            detection_pos = tf.apply(measurement)
            if translate:
                detection_pos += np.asarray(trans)
            return detection_pos
        except:
            return np.array([np.nan])"""

    def update_daemon_service(self, req):
        data_frame = req.objdets

        # transform detections to world frame
        """for datum in data_frame:
            pos_in_world = self.transform_meas_to_world(point_to_array(datum.position), \
                                                        datum.header.frame_id, "odom", datum.header.stamp)
            datum.position = array_to_point(pos_in_world)
            datum.header.frame_id = "odom"
        """
        
        # acquire and update data buffer on daemon
        daemon_name = req.detector_tag
        if daemon_name in self.daemon_dict:
            daemon = self.daemon_dict[daemon_name]
            daemon.mutex.acquire()
            daemon.update_detection_buffer(data_frame)
            daemon.mutex.release()
            return True
        return False

    #iterate through all daemons and call spin function to update tracking
    def spin(self, event):
        for daemon_name in self.daemon_dict:
            daemon = self.daemon_dict[daemon_name]
            daemon.mutex.acquire()
            daemon.spin()
            daemon.mutex.release()

def main():
    rospy.init_node('detector_bucket', anonymous=True)
    detector_bucket = Detector_Bucket()
    rospy.spin()




