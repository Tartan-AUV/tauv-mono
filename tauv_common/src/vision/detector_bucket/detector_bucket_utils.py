# detector_bucket_utils
#
# Provides all the libraries necessary for carrying out data association from the detector_bucket
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
from tauv_msgs.msg import BucketDetection, BucketList, PoseGraphMeasurement
from tauv_common.srv import RegisterObjectDetections, RegisterMeasurement
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv, block_diag
from threading import Thread, Lock
from scipy.optimize import linear_sum_assignment


def array_to_point(arr):
    p = Point()
    p.x = arr[0]
    p.y = arr[1]
    p.z = arr[2]
    return p

def point_to_array(point):
    return np.asarray([point.x, point.y, point.z])

# constant position Kalman Filter for stationary object tracking
class Detection_Tracker_Kalman():
    def __init__(self):
        self.id = -1
        self.state_space_dim = 3
        self.localized_point = np.zeros((self.state_space_dim, 1))
        self.detections = 0
        self.estimated_point = np.zeros((self.state_space_dim, 1))
        self.last_updated_time = 0.0

        # Process matrix
        self.F = np.asmatrix([[1, 0, 0], # x' = x
                              [0, 1, 0], # y' = y
                              [0, 0, 1]])  # z' = z

        # Measurement matrix
        # measurement is reported as [x, y, z] in world frame
        self.H = np.asmatrix([[1, 0, 0],   # x = zx
                              [0, 1, 0],   # y = zy
                              [0, 0, 1]])  # z = zz

        # State covariance, initialize to very large number
        self.P = 100000*np.eye(3)

        # Process noise covariance
        self.Q = .0001*np.eye(3)

        # Measurement covariance (x,y,z), assume the sensor is very noisy
        self.R = 100000*np.eye(3)

        #need a override covariance for when no measurements are available
        self.R_override = (10**10)*np.eye(3)


    # measurement is of the form [x, y, z] in the world frame
    def kalman_predict_and_update(self, measurement, override=False):
        x = self.estimated_point
        #print(measurement)
        #print(measurement.shape)
        # Kalman Predict
        x = self.F * x # predict state
        self.P = self.F * self.P * self.F.T + self.Q # predict state cov

        #Kalman Update
        if override:
            S = self.H * self.P * self.H.T + self.R_override
        else:
            S = self.H * self.P * self.H.T + self.R

        K = self.P * self.H.T * inv(S) #Kalman gain
        #print(K.shape)
        y = measurement - self.H * x #Residual
        #print(x.shape)
        #print((K*y).shape)
        x += K * y
        self.P = self.P - K * self.H * self.P
        self.estimated_point = x

    def kalman_predict(self):
        x = self.estimated_point
        x = self.F * x
        self.P = self.F * self.P * self.F.T + self.Q
        self.estimated_point = x

    #need to update
    def update_dt(self, new_measurement_time):
        self.dt = new_measurement_time - self.last_updated_time
        self.last_updated_time = new_measurement_time

class Detector_Daemon():
    def __init__(self, detector_name, daemon_id):
        self.detector_name = detector_name
        self.daemon_id = daemon_id
        self.mutex = Lock()

        #each daemon will call the service to report a measurement
        rospy.wait_for_service("/gnc/pose_graph/register_measurement")
        self.meas_reg_service = rospy.ServiceProxy("/gnc/pose_graph/register_measurement", RegisterMeasurement)
        self.mahalanobis_threshold = 1
        if rospy.has_param("detectors/" + self.detector_name + "/mahalanobis_threshold"):
            rospy.loginfo("[Detector Daemon]: %s. Obtained mahalanobis threshold", self.detector_name)
            self.mahalanobis_threshold = float(rospy.get_param("detectors/" + self.detector_name + "/mahalanobis_threshold"))
        self.marker_pub = rospy.Publisher(self.detector_name + "_daemon/filtered_det_marker", MarkerArray, queue_size=10)

        #list of detections for this daemon
        self.detection_buffer = []
        self.new_data = False

        self.debouncing_threshold = 10
        if rospy.has_param("detectors/" + self.detector_name + "/debouncing_threshold"):
            rospy.loginfo("[Detector Daemon]: %s. Obtained debouncing threshold", self.detector_name)
            self.debouncing_threshold = float(rospy.get_param("detectors/" + self.detector_name + "/debouncing_threshold"))
        self.tracker_list = []
        self.trackers_to_publish = {}

        self.tracker_id = 0
        self.marker_dict = {}

    def update_detection_buffer(self, data_frame):
        #rospy.loginfo("Updated detection buffer in %s daemon", self.detector_name)
        self.detection_buffer.append(data_frame)
        self.new_data = True

    # performs assignment to trackers and detections
    # returns (matches, unmatched detections, unmatched trackers)
    def map_det_to_tracker(self, trackers, dets):
        #rospy.loginfo("Finding map function in %s daemon", self.detector_name)
        trackers_len = len(trackers)
        dets_len = len(dets)

        trackers_tiled = np.tile(trackers, (dets_len, 1))
        dets_repeated = np.repeat(dets, trackers_len, axis=0)
        if trackers_len < 1:
            tracker_matches, det_matches = np.array([]), np.array([])
        else:
            adjacency_cost_matrix = np.reshape(np.linalg.norm(trackers_tiled - dets_repeated, axis=1), (trackers_len, dets_len))
            tracker_matches, det_matches = linear_sum_assignment(adjacency_cost_matrix)

        unmatch_tracks, unmatch_dets = [], []
        whole_trackers = set(list(range(trackers_len)))
        whole_dets = set(list(range(dets_len)))
        unmatch_tracks = list(whole_trackers - set(tracker_matches))
        unmatch_dets = list(whole_dets - set(det_matches))

        matches = []
        for match in range(len(tracker_matches)):
            i = tracker_matches[match]
            j = det_matches[match]
            if adjacency_cost_matrix[i, j] < self.mahalanobis_threshold:
                matches.append(np.array([i, j]))
            else:
                unmatch_tracks.append(i)
                unmatch_dets.append(j)

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        elif len(matches) == 1:
            matches = np.asmatrix(matches)
        else:
            matches = np.stack(matches,axis=0)

        return matches, np.array(unmatch_dets), np.array(unmatch_tracks)

    def override_localization(self, dets):
        for det in dets:
            tag = det.tag
            det_in_world = point_to_array(det.position)
            if rospy.has_param(tag + "/location_override_x"):
                override = float(rospy.get_param(tag + "/location_override_x"))
                det_in_world[0] = override
            if rospy.has_param(tag + "/location_override_y"):
                override = float(rospy.get_param(tag + "/location_override_y"))
                det_in_world[1] = override
            if rospy.has_param(tag + "/location_override_z"):
                override = float(rospy.get_param(tag + "/location_override_z"))
                det_in_world[2] = override
            det.position = array_to_point(det_in_world)

    # performs association and updates Kalman trackers, then publishes them to pose graph
    def spin(self):
        #rospy.loginfo("In Daemon spin")
        if self.new_data:
            while len(self.detection_buffer) > 0:
                # gather data
                data_frame = self.detection_buffer.pop(0)
                if len(data_frame) > 0:
                    self.override_localization(data_frame)
                    measurements = np.asmatrix([point_to_array(datum.position) for datum in data_frame])
                    #print("Meas:" + str(measurements))
                    time_stamp = data_frame[0].header.stamp

                    # match trackers and detections
                    #print(self.tracker_list)
                    temp_tracker_holder = [tracker.localized_point for tracker in self.tracker_list]
                    #print("Tmp:" + str(temp_tracker_holder))
                    matches, unmatch_dets, unmatch_tracks = \
                        self.map_det_to_tracker(temp_tracker_holder, measurements)
                    #print(matches, unmatch_dets, unmatch_tracks)
                    if matches.size > 0:
                        for ii in range(len(matches)):
                            print(("MATCHED in %s" + str(matches[ii])), self.detector_name)
                            #print("INE 1")
                            #print(matches[ii])
                            tracker_ind = matches[ii, 0]
                            detection_ind = matches[ii, 1]
                            measurement = measurements[detection_ind]
                            #print("MEAS: " + str(measurement) + "SIZE" + str(measurement.shape))
                            measurement = np.asmatrix(measurement).T
                            #print(measurement.shape)
                            #kalman update and predict
                            tracker = self.tracker_list[tracker_ind]
                            tracker.kalman_predict_and_update(measurement)
                            new_x = tracker.estimated_point.T[0].tolist()
                            #print("new_x:" + str(new_x[0]))
                            temp_tracker_holder[tracker_ind] = new_x[0]
                            tracker.localized_point = new_x[0]
                            tracker.detections += 1

                    if len(unmatch_dets) > 0:
                        #print("INE 2")
                        for detection_ind in unmatch_dets:
                            measurement = measurements[detection_ind]
                            measurement = measurement.T
                            #print(measurement)

                            #create new tracker
                            #print("NEW TRACKER: " + str(measurement))
                            new_tracker = Detection_Tracker_Kalman()
                            new_tracker.estimated_point = np.asmatrix(measurement)
                            #print(new_tracker.estimated_point.shape)
                            new_tracker.kalman_predict()

                            #update state and id
                            new_x = new_tracker.estimated_point
                            new_x = new_x.T[0].tolist()
                            #print("new_x:" + str(new_x[0]))
                            new_tracker.localized_point = new_x[0]
                            new_tracker.id = self.tracker_id

                            #add to list
                            self.tracker_id += 1
                            self.tracker_list.append(new_tracker)
                            temp_tracker_holder.append(new_x[0])

                    if len(unmatch_tracks) > 0:
                        print("INE 3")
                        for tracker_ind in unmatch_tracks:
                            tracker = self.tracker_list[tracker_ind]
                            #print("LOST TRACKLER" + str(tracker_ind) + str(tracker.estimated_point))
                            #override the update
                            tracker.kalman_predict_and_update(np.asmatrix([0, 0, 0]).T, True)
                            new_x = tracker.estimated_point
                            #print(new_x.shape)
                            new_x = new_x.T[0].tolist()
                            tracker.localized_point = new_x[0]
                            #print("new_x:" + str(new_x[0]))
                            temp_tracker_holder[tracker_ind] = new_x[0]

                    trackers_to_be_published = []
                    for tracker in self.tracker_list:
                        if tracker.detections >= self.debouncing_threshold:
                            trackers_to_be_published.append(tracker)

                    self.trackers_to_publish = {tracker.id: (time_stamp, tracker.localized_point) for tracker in trackers_to_be_published}
                    for tracker in self.trackers_to_publish:
                        pos = self.trackers_to_publish[tracker][1]
                        self.create_marker(pos, tracker)
                    self.marker_pub.publish(self.marker_dict.values())
                    #print(self.trackers_to_publish)
                    #TODO: transfrom all the detections back into the sensor frame before publishing to pose_graph
            self.new_data = False

    def create_marker(self, pos, id):
        m = Marker()
        m.header.frame_id = "odom"
        m.id = id
        m.type = 1
        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = pos[2]
        m.color.g = 1.0
        m.color.a = 1.0
        m.scale.x = .5
        m.scale.y = .5
        m.scale.z = .5
        self.marker_dict[id] = m






    # # transform a 3D measurement from child frame to world frame
    # def transform_meas_to_world(self, measurement, child_frame, world_frame, time):
    #     try:
    #         (trans, rot) = self.tf.lookupTransform(world_frame, child_frame, time)
    #         tf = R.from_quat(np.asarray(rot))
    #         detection_pos = tf.apply(measurement) + np.asarray(trans)
    #         return detection_pos
    #     except:
    #         return np.array([np.nan])


    # def find_nearest_neighbor(self, bucket_detection):
    #     if len(self.bucket_dict.keys()) > 0:
    #         curr_det_positions = [self.bucket_dict[id][0].position for id in self.bucket_dict]
    #         curr_det_positions = np.asarray(list(map(self.point_to_array, curr_det_positions))).T
    #         new_det_position = np.asarray(np.asarray([bucket_detection.position.x, \
    #                                                   bucket_detection.position.y, bucket_detection.position.z])).T
    #         diff = np.asmatrix(new_det_position[:, None] - curr_det_positions)
    #         mahalanobis_distance = np.sqrt(np.diag(diff.T*np.diag([.3, .3, .3])*diff)) #replace with inverse covariance matrix
    #         # #print("curr:" + str(curr_det_positions))
    #         # #print("new:" + str(new_det_position))
    #         # #print("Maha: "+ str(mahalanobis_distance))
    #         nearest_neighbor = np.argmin(mahalanobis_distance)
    #         # #print("[Debounced Detection Tracker]: " + str(self.debouncing_tracker_dict))
    #         tag = bucket_detection.tag
    #
    #         if mahalanobis_distance[nearest_neighbor] < self.nn_threshold: #new detection is already seen by system
    #             self.debouncing_tracker_dict[nearest_neighbor] += 1
    #             return nearest_neighbor
    #     self.monotonic_det_id += 1
    #     return self.monotonic_det_id


