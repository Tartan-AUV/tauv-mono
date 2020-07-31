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

def array_to_point(self, arr):
    p = Point()
    p.x = arr[0]
    p.y = arr[1]
    p.z = arr[2]
    return p

def point_to_array(self, point):
    return np.asarray([point.x, point.y, point.z])

class Detection_Tracker_Kalman():
    def __init__(self):
        self.id = 0
        self.state_space_dim = 3
        self.localized_point = np.zeros(self.state_space_dim, 1)
        self.detections = 0
        self.estimated_point = np.zeros(self.state_space_dim, 1)
        self.last_updated_time = 0.0

        # Process matrix
        self.F = np.asmatrix([[1, 0, 0], # x' = x
                              [0, 1, 0], # y' = y
                              [0, 0, 1]])  # z' = z

        # Measurement matrix
        # measurement is reported as [x, y, z] in world frame
        self.H = np.asmatrix([[1, 0, 0],   # x
                              [0, 1, 0],   # y
                              [0, 0, 1]])  # z

        # State covariance, initialize to very large number
        self.P = 10000*np.eye(3)

        # Process noise covariance
        self.Q = .0001*np.eye(3)

        # Measurement covariance (x,y,z), assume the sensor is very noisy
        self.R = 10000*np.eye(3)

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

        #list of detections for this daemon
        self.detection_buffer = []
        self.new_data = False

        self.debouncing_threshold = 3
        self.tracker_list = []
        self.mahalanobis_threshold = .3
        self.tracker_id = 0

    def update_detection_buffer(self, data_frame):
        self.detection_buffer.append(data_frame)

    # performs assignment to trackers and detections
    # returns (matches, unmatched detections, unmatched trackers)
    def map_det_to_tracker(self, trackers, dets):
        return

    # performs association and updates Kalman trackers, then publishes them to pose graph
    def spin(self):
        if self.new_data:
            self.new_data = False
            while len(self.detection_buffer) > 0:
                # gather data
                data_frame = self.detection_buffer.pop(0)
                measurements = [point_to_array(datum.position) for datum in data_frame]
                time_stamp = data_frame[0].header.stamp

                # match trackers and detections
                temp_tracker_holder = [tracker.localized_point for tracker in self.tracker_list]
                matches, unmatch_dets, unmatch_tracks = \
                    self.map_det_to_tracker(temp_tracker_holder, measurements)

                if matches.size() > 0:
                    for tracker_ind, detection_ind in matches:
                        measurement = measurements[detection_ind]
                        measurement = np.expand_dims(measurement, axis=0).T
                        tracker = self.tracker_list[tracker_ind]
                        tracker.kalman_predict_and_update(measurement)
                        new_x = tracker.estimated_point.T[0].tolist()
                        temp_tracker_holder[tracker_ind] = new_x
                        tracker.localized_point = new_x
                        tracker.detections += 1

                if len(unmatch_dets) > 0:
                    for detection_ind in unmatch_dets:
                        measurement = measurements[detection_ind]
                        measurement = np.expand_dims(measurement, axis=0).T
                        new_tracker = Detection_Tracker_Kalman()
                        new_tracker.estimated_point = measurement
                        new_tracker.kalman_predict()
                        new_x = new_tracker.estimated_point
                        new_x = new_x.T[0].tolist()
                        new_tracker.localized_point = new_x
                        new_tracker.id = self.tracker_id
                        self.tracker_id += 1
                        self.tracker_list.append(new_tracker)
                        temp_tracker_holder.append(new_x)

                if len(unmatch_tracks) > 0:
                    for tracker_ind in unmatch_tracks:
                        tracker = self.tracker_list[tracker_ind]
                        tracker.kalman_predict_and_update(np.asarray([0, 0, 0]), True)



    # transform a 3D measurement from child frame to world frame
    def transform_meas_to_world(self, measurement, child_frame, world_frame, time):
        try:
            (trans, rot) = self.tf.lookupTransform(world_frame, child_frame, time)
            tf = R.from_quat(np.asarray(rot))
            detection_pos = tf.apply(measurement) + np.asarray(trans)
            return detection_pos
        except:
            return np.array([np.nan])


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


