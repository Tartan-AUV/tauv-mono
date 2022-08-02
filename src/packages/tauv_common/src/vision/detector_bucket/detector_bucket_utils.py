# detector_bucket_utils
#
# Provides all the libraries necessary for carrying out data association from the detector_bucket
# Contains a 3-D constant position model Kalman Filter Tracker
# Contains a generic detector Daemon, each is assigned to one sensor to keep track of its detections
# This allows asynchronous sensor updates to the pose graph, and will reduce bottlenecking
# Everything is done here in the world frame, as data association needs to happen in a fixed frame
#
# Author: Advaith Sethuraman 2020


#!/usr/bin/env python
import rospy
#import tf
#import tf_conversions
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
from geometry_msgs.msg import Quaternion, Point
from tauv_msgs.msg import BucketDetection, BucketList, PoseGraphMeasurement,RegisterObjectDetections, RegisterMeasurement
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv, block_diag
from threading import Thread, Lock
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import typing

tracker_id = 0

def array_to_point(arr):
    p = Point()
    p.x = arr[0]
    p.y = arr[1]
    p.z = arr[2]
    return p

def point_to_array(point):
    return np.asarray([point.x, point.y, point.z])

# constant position Kalman Filter for stationary object tracking
class Detection_Tracker_Kalman(BucketDetection):
    def __init__(self, tag):
        self.id = -1
        self.updated_now = False

        self.state_space_dim = 3
        self.localized_point = np.zeros((self.state_space_dim, 1))
        self.detections = 0

        self.tag = tag
        self.estimated_point = np.zeros((self.state_space_dim, 1))
        self.last_updated_time = 0.0

        self.position = Point(0,0,0)
        self.header = Header()

        self.count = 1

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
        self.P = 10000*np.eye(3)

        # Process noise covariance
        self.Q = .1*np.eye(3)

        # Measurement covariance (x,y,z), assume the sensor is very noisy
        self.R = 100*np.eye(3)

        #need a override covariance for when no measurements are available
        self.R_override = (10**10)*np.eye(3)


    # measurement is of the form [x, y, z] in the world frame
    def kalman_predict_and_update(self, measurement, override=False):
        x = self.estimated_point

        # Kalman Predict
        x = self.F * x # predict state
        self.P = self.F * self.P * self.F.T + self.Q # predict state cov

        #Kalman Update
        if override:
            S = self.H * self.P * self.H.T + self.R_override
        else:
            S = self.H * self.P * self.H.T + self.R

        K = self.P * self.H.T * inv(S) #Kalman gain
        y = measurement - self.H * x #Residual
        x += K * y
        self.P = self.P - K * self.H * self.P
        self.estimated_point = x
        self.position = Point(x[0][0],x[1][0],x[2][0])

    def kalman_predict(self):
        x = self.estimated_point
        x = self.F * x
        self.P = self.F * self.P * self.F.T + self.Q
        self.estimated_point = x
        self.position = array_to_point(x)

    def update_dt(self, new_measurement_time):
        self.dt = new_measurement_time - self.last_updated_time
        self.last_updated_time = new_measurement_time

class Detector_Daemon(BucketList):
    def __init__(self, detector_name, daemon_id):
        self.detector_name = detector_name
        self.daemon_id = daemon_id
        self.mutex = Lock()
        self.header = Header()

        #used to see how many of a particular object were identified
        self.track_map = dict()
        self.max_map = dict()

        #each daemon will call the service to report a measurement
        self.mahalanobis_threshold = 1
        if rospy.has_param("detectors/" + self.detector_name + "/mahalanobis_threshold"):
            rospy.loginfo("[Detector Daemon]: %s. Obtained mahalanobis threshold", self.detector_name)
            self.mahalanobis_threshold = float(rospy.get_param("detectors/" + self.detector_name + "/mahalanobis_threshold"))


        #list of detections for this daemon
        self.detection_buffer = []
        self.new_data = False

        self.debouncing_threshold = 10
        if rospy.has_param("detectors/" + self.detector_name + "/debouncing_threshold"):
            rospy.loginfo("[Detector Daemon]: %s. Obtained debouncing threshold", self.detector_name)
            self.debouncing_threshold = float(rospy.get_param("detectors/" + self.detector_name + "/debouncing_threshold"))
        
        self.tracker_list: typing.List[BucketDetection] = []
        self.bucket_list = []

        self.frame_id = "odom"
        if rospy.has_param("detectors/" + self.detector_name + "/frame_id"):
            rospy.loginfo("[Detector Daemon]: %s. Obtained detector frame_id", self.detector_name)
            self.frame_id = str(rospy.get_param("detectors/" + self.detector_name + "/frame_id"))
        else:
            rospy.logerr("[Detector Daemon]: %s: Detector frame_id not specified!" % self.detector_name)

        self.tracker_id = 0

    def reset(self):
        self.mutex.acquire()
        self.tracker_list = []
        self.bucket_list = []
        self.track_map = {}
        self.max_map = {}
        self.mutex.release()

    def update_detection_buffer(self, data_frame):
        #rospy.loginfo("Updated detection buffer in %s daemon", self.detector_name)
        #rospy.loginfo(f"DATAFRAM: {data_frame}")
        self.detection_buffer.append(data_frame)
        self.new_data = True

    # performs assignment to trackers and detections
    def map_det_to_tracker(self, trackers, dets, tracker_tags, det_tags):
        trackers_len = len(trackers)
        dets_len = len(dets)

        trackers_tiled = np.tile(trackers, (dets_len, 1))
        dets_repeated = np.repeat(dets, trackers_len, axis=0)
        if trackers_len > 0 and dets_len > 0 and \
            not (np.any(np.isinf(trackers)) | np.any(np.isnan(trackers)) \
             |  np.any(np.isinf(dets)) | np.any(np.isnan(dets))):
            adjacency_cost_matrix = np.reshape(np.linalg.norm(trackers_tiled - dets_repeated, axis=1), \
                                               (trackers_len, dets_len))
            tracker_matches, det_matches = linear_sum_assignment(adjacency_cost_matrix)
        else:
            tracker_matches, det_matches = np.array([]), np.array([])


        unmatch_tracks, unmatch_dets = [], []
        whole_trackers = set(list(range(trackers_len)))
        whole_dets = set(list(range(dets_len)))
        unmatch_tracks = list(whole_trackers - set(tracker_matches))
        unmatch_dets = list(whole_dets - set(det_matches))

        matches = []
        for match in range(len(tracker_matches)):
            i = tracker_matches[match]
            j = det_matches[match]
            if(adjacency_cost_matrix[i, j] < self.mahalanobis_threshold and tracker_tags[i]==det_tags[j]):
                matches.append(np.array([i, j]))
            #elif(self.track_map[det_tags[j]]!=self.max_map[det_tags[j]]):
                #rospy.loginfo(f"cost: {adjacency_cost_matrix[i, j]}")
                #rospy.loginfo(f"adjacency matrix: {adjacency_cost_matrix}")
            else:
                unmatch_tracks.append(i)
                unmatch_dets.append(j)

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        elif len(matches) == 1:
            matches = np.asmatrix(matches)
        else:
            matches = np.stack(matches,axis=0)

        #rospy.loginfo(f"matches: {matches.size()} unmatched: {unmatch_dets.size()}")

        return matches, np.array(unmatch_dets), np.array(unmatch_tracks)


    # incorporates any priors about the landmark position
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

    # performs association, finds matches and updates Kalman trackers, then publishes them to pose graph
    def spin(self):
        if self.new_data:
            while len(self.detection_buffer) > 0:
                data_frame = self.detection_buffer.pop(0)
                if len(data_frame) > 0:
                    self.override_localization(data_frame)

                    temp = []
                    det_tags = []

                    for datum in data_frame:
                        #rospy.loginfo(f"POSITION: {datum.position}")
                        temp.append(point_to_array(datum.position))
                        det_tags.append(datum.tag)

                    measurements = np.asmatrix(temp)

                    time_stamp = data_frame[0].header.stamp

                    # match trackers and detections
                    temp_tracker_holder = []
                    tracker_tags = []

                    for tracker in self.tracker_list:
                        tracker_tags.append(tracker.tag)
                        temp_tracker_holder.append(tracker.localized_point)


                    matches, unmatch_dets, unmatch_tracks = self.map_det_to_tracker(temp_tracker_holder, measurements, tracker_tags, det_tags)

                    #matches, unmatch_dets, unmatch_tracks = self.match_by_euclidian(self.bucket_list, data_frame)


                    if matches.size > 0:
                        for ii in range(len(matches)):
                            tracker_ind = matches[ii,0]
                            detection_ind = matches[ii,1]
                            measurement = measurements[detection_ind]
                            measurement = np.asmatrix(measurement).T

                            #kalman update and predict
                            tracker = self.tracker_list[tracker_ind]
                            tracker.kalman_predict_and_update(measurement)
                            tracker.count+=1
                            new_x = tracker.estimated_point.T[0].tolist()

                            temp_tracker_holder[tracker_ind] = new_x[0]
                            tracker.localized_point = new_x[0]
                            tracker.count = min(100, tracker.count + 1)

                            self.track_map[tracker.tag] += 1
                            for i in range(len(self.tracker_list)):
                                if i != tracker_ind and self.tracker_list[i].count > 5:
                                    self.tracker_list[i].count -= 2

                            tracker.updated_now = True

                    if len(unmatch_dets) > 0:
                        for detection_ind in unmatch_dets:
                            measurement = measurements[detection_ind]
                            measurement = measurement.T

                            rospy.loginfo("[Detector Daemon]: Unmatched: " + str(measurement))

                            #create new tracker
                            tag = data_frame[detection_ind].tag
                            new_tracker = Detection_Tracker_Kalman(tag)
                            new_tracker.estimated_point = np.asmatrix(measurement)
                            new_tracker.kalman_predict()

                            #update state and id
                            new_x = new_tracker.estimated_point
                            new_x = new_x.T[0].tolist()
                            new_tracker.localized_point = new_x[0]

                            global tracker_id
                            new_tracker.id = self.tracker_id

                            #add to list
                            self.tracker_id += 1
                            self.tracker_list.append(new_tracker)
                            temp_tracker_holder.append(new_x[0])

                            self.track_map[new_tracker.tag] = 1
                            self.max_map[new_tracker.tag] = 2

                            if(rospy.has_param("object_tags/" + new_tracker.tag +"/total_number")):
                                self.max_map[new_tracker.tag] = int(rospy.get_param("object_tags/" + new_tracker.tag +"/total_number"))

                    if len(unmatch_tracks) > 0:
                        for tracker_ind in unmatch_tracks:
                            tracker = self.tracker_list[tracker_ind]

                            #override the update
                            tracker.kalman_predict_and_update(np.asmatrix([0, 0, 0]).T, True)
                            new_x = tracker.estimated_point

                            new_x = new_x.T[0].tolist()
                            tracker.localized_point = new_x[0]
                            temp_tracker_holder[tracker_ind] = new_x[0]

                    self.bucket_list = self.tracker_list


            self.new_data = False