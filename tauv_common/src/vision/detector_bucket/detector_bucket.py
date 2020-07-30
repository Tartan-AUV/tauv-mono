# detector_bucket
#
# This node is the for aggregating the detections from the vision pipeline.
# Input: Detection
#
# Author: Advaith Sethuraman 2020


#!/usr/bin/env python
import rospy
import tf
import tf_conversions
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
from tauv_common.srv import RegisterObjectDetection, RegisterMeasurement
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R

class Detector_Bucket():
    def __init__(self):
        self.bucket_list_pub = rospy.Publisher("bucket_list", BucketList, queue_size=50)
        self.bbox_3d_list_pub = rospy.Publisher("bucket_bbox_3d_list", BoundingBoxArray, queue_size=50)
        self.detection_server = rospy.Service("detector_bucket/register_object_detection", RegisterObjectDetection, \
                                              self.register_object_detection)
        self.arrow_pub = rospy.Publisher("detection_marker", MarkerArray, queue_size=10)
        self.num_daemons = 1
        self.daemon_names = ["default"]
        self.daemon_dict = {self.daemon_names[0]: Detector_Daemon(self.daemon_names[0])}

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
        rospy.wait_for_service("/gnc/pose_graph/register_measurement")
        self.meas_reg_service = rospy.ServiceProxy("/gnc/pose_graph/register_measurement", RegisterMeasurement)
        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)

    def init_daemons(self):
        if rospy.has_param("detectors/total_number"):
            self.num_daemons = int(rospy.get_param("detectors/total_number"))
            self.daemon_names = rospy.get_param("detectors/names")
        self.daemon_dict = {name: Detector_Daemon(name) for name in self.daemon_names}
        

    def is_valid_registration(self, new_detection):
        tag = new_detection.tag != ""
        return tag

    def transform_meas_to_world(self, measurement, child_frame, world_frame, time):
        try:
            (trans, rot) = self.tf.lookupTransform(world_frame, child_frame, time)
            tf = R.from_quat(np.asarray(rot))
            detection_pos = tf.apply(measurement) + np.asarray(trans)
            return detection_pos
        except:
            return np.array([np.nan])

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

    # returns nearest landmark neighbor in bucket or -1 to signify a new marker
    # Eventually include CBOW and prior object information
    def find_nearest_neighbor(self, bucket_detection):
        if len(self.bucket_dict.keys()) > 0:
            curr_det_positions = [self.bucket_dict[id][0].position for id in self.bucket_dict]
            curr_det_positions = np.asarray(list(map(self.point_to_array, curr_det_positions))).T
            new_det_position = np.asarray(np.asarray([bucket_detection.position.x, \
                                                      bucket_detection.position.y, bucket_detection.position.z])).T
            diff = np.asmatrix(new_det_position[:, None] - curr_det_positions)
            mahalanobis_distance = np.sqrt(np.diag(diff.T*np.diag([.3, .3, .3])*diff)) #replace with inverse covariance matrix
            # print("curr:" + str(curr_det_positions))
            # print("new:" + str(new_det_position))
            # print("Maha: "+ str(mahalanobis_distance))
            nearest_neighbor = np.argmin(mahalanobis_distance)
            # print("[Debounced Detection Tracker]: " + str(self.debouncing_tracker_dict))
            tag = bucket_detection.tag

            if mahalanobis_distance[nearest_neighbor] < self.nn_threshold: #new detection is already seen by system
                self.debouncing_tracker_dict[nearest_neighbor] += 1
                return nearest_neighbor
        self.monotonic_det_id += 1
        return self.monotonic_det_id

    def array_to_point(self, arr):
        p = Point()
        p.x = arr[0]
        p.y = arr[1]
        p.z = arr[2]
        return p

    def point_to_array(self, point):
        return np.asarray([point.x, point.y, point.z])

    def register_object_detection(self, req):
        bucket_detection = req.objdet
        bbox_3d_detection = bucket_detection.bbox_3d
        tag = bucket_detection.tag

        if self.is_valid_registration(bucket_detection):
            now = bucket_detection.header.stamp
            child_frame = bucket_detection.header.frame_id

            #transform into odom and update the detections (temporary, will be published by SLAM backend in odom frame)
            pos = self.point_to_array(bucket_detection.position)
            det_in_world = self.transform_meas_to_world(pos, child_frame, "odom", now)
            if not np.any(np.isnan(det_in_world)):
                if rospy.has_param(tag + "/location_override_x"):
                    override = float(rospy.get_param(tag + "/location_override_x"))
                    det_in_world[0] = override
                if rospy.has_param(tag + "/location_override_y"):
                    override = float(rospy.get_param(tag + "/location_override_y"))
                    det_in_world[1] = override
                if rospy.has_param(tag + "/location_override_z"):
                    override = float(rospy.get_param(tag + "/location_override_z"))
                    det_in_world[2] = override

                det_in_world = self.array_to_point(det_in_world)
                bucket_detection.position = det_in_world
                bbox_3d_detection.pose.position = det_in_world
                bucket_detection.header.frame_id = "odom"
                bbox_3d_detection.header.frame_id = "odom"

                #find nearest neighbor, or add new detection
                det_id = self.find_nearest_neighbor(bucket_detection)

                #always add new detections to the bucket_dict, debouncing dict is filtered output
                if det_id not in self.debouncing_tracker_dict: #new detection
                    self.debouncing_tracker_dict[det_id] = 1
                self.bucket_dict[det_id] = (bucket_detection, bbox_3d_detection)

                total_number = float('Inf')
                if rospy.has_param(tag + "/total_number"):
                    total_number = float(rospy.get_param(tag + "/total_number"))

                #only allow detections that persisted for threshold to enter the detections for a time frame
                if self.debouncing_tracker_dict[det_id] > self.debouncing_threshold:
                    #and self.total_number_detection_dict.get(bucket_detection.tag, 0) <= total_number:
                    self.debounced_detection_dict[det_id] = (bucket_detection, bbox_3d_detection)
                    if bucket_detection.tag not in self.total_number_detection_dict:
                        self.total_number_detection_dict[bucket_detection.tag] = 1
                    else:
                        self.total_number_detection_dict[bucket_detection.tag] += 1
                return True
        return False

    #publish new detections for time stamp to SLAM backend
    def spin(self, event):
        print("IN SPIN")
        now = rospy.Time(0)
        if len(self.debounced_detection_dict.keys()) > 0:

            #update all the arrows for visualization
            for dd_id in self.debounced_detection_dict:
                d_det = self.debounced_detection_dict[dd_id][0]
                ts = d_det.header.stamp
                child_frame = d_det.header.frame_id
                robot_in_world = self.array_to_point(self.transform_meas_to_world(np.asarray([0, 0, 0]), "base_link", "odom", ts))
                self.update_detection_arrows(d_det, "odom", robot_in_world, dd_id)
                pg_meas = PoseGraphMeasurement()
                pg_meas.header = d_det.header
                pg_meas.header.stamp = ts
                pg_meas.position = d_det.position
                pg_meas.landmark_id = dd_id
                success = self.meas_reg_service(pg_meas)

            bucket_list_msg = BucketList()
            bbox_3d_list_msg = BoundingBoxArray()
            bucket_list_msg.header = Header()
            bucket_list_msg.bucket_list = [self.debounced_detection_dict[id][0] for id in self.debounced_detection_dict]
            bbox_3d_list_msg.header = Header()
            bbox_3d_list_msg.header.frame_id = "odom"
            bbox_3d_list_msg.boxes = [self.debounced_detection_dict[id][1] for id in self.debounced_detection_dict]

            self.bucket_list_pub.publish(bucket_list_msg)
            self.bbox_3d_list_pub.publish(bbox_3d_list_msg)
            self.arrow_pub.publish(self.arrow_dict.values())

        return

def main():
    rospy.init_node('detector_bucket', anonymous=True)
    detector_bucket = Detector_Bucket()
    rospy.spin()




