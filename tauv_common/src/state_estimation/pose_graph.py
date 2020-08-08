# pose_graph
#
# This class will house the pose graph interface
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
from tauv_common.srv import RegisterMeasurement
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
from vision.detector_bucket.detector_bucket_utils import *


from threading import Thread, Lock
import torch

class Pose_Graph_Edge():
    def __init__(self, type, parent_id, child_id):
        self.parent_id = parent_id
        self.child_id = child_id
        self.transform = tf

class Pose_Graph_Node():
    def __init__(self, node_id, pose, type):
        self.node_id = node_id
        self.state = Pose()
        self.type = type #pose or landmark
        self.factor_relations = {"landmarks":[], "poses":[]} #contains tuples of (node_id, spatial relation)

    def add_features(self):
        return

    def get_features(self):
        return

    def get_transform(self, other_node):
        return

    def set_pose_from_transform(self, parent_pg_node, transform):
        return

    def add_neighbor(self, type, node_id):
        return

class Pose_Graph():
    def __init__(self):
        self.nodes = {"landmarks":dict(), "poses":dict()}
        self.tf = tf.TransformListener()
        self.marker_pub = rospy.Publisher("pose_marker", MarkerArray, queue_size=10)
        self.detection_pub = rospy.Publisher("detection_marker", MarkerArray, queue_size=10)
        self.detection_marker_dict = {}
        self.marker_dict = {}
        self.detections_dict = {}
        self.data_mutex = Lock()
        self.color_cycle = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        self.flip = 0
        self.marker_id = 0
        self.new_measurement = False
        self.measurement_server = rospy.Service("pose_graph/register_measurement", RegisterMeasurement, \
                                              self.register_measurement)

        #begin test trajectory
        #


        self.prev_state = torch.zeros(0)
        self.got_prev = False
        self.state_dim = 7
        self.meas_dim = 3
        self.landmark_offset = 0
        self.num_poses = 0
        self.num_landmarks = 0
        self.time_step = 0
        self.clear_mark = Marker()
        self.clear_mark.action = 3

        self.indices = torch.LongTensor()
        self.values = torch.FloatTensor()
        self.J = torch.sparse.FloatTensor()
        self.virtual_obs = torch.FloatTensor() # use estimate to calculate virtual obs
        self.real_obs = torch.FloatTensor() # take difference in poses, take actual landmark measurements
        self.estimate = torch.FloatTensor() # has state_dim*num_poses + meas_dum*num_landmarks values
        #offset @ state_dim*num_poses

        self.marker_callback = rospy.Timer(rospy.Duration(.010), self.spin)

    def transform_meas_to_frame(self, measurement, child_frame, world_frame, time):
        try:
            (trans, rot) = self.tf.lookupTransform(world_frame, child_frame, time)
            tf = R.from_quat(np.asarray(rot))
            detection_pos = tf.apply(measurement) + np.asarray(trans)
            return detection_pos
        except:
            return np.array([np.nan])

    def register_measurement(self, req):
        self.data_mutex.acquire()
        self.detections_dict = {}
        for datum in req.pg_measurements:
            id = datum.landmark_id
            frame_id = datum.header.frame_id
            pos = self.transform_meas_to_frame(point_to_array(datum.position), frame_id, "odom", datum.header.stamp)
            self.detections_dict[id] = (pos, frame_id)
        self.data_mutex.release()
        return True

    def get_current_state(self):
        try:
            (trans, rot) = self.tf.lookupTransform(world_frame, child_frame, time)
            return np.concatenate([trans, rot])
        except:
            return np.array(self.state_dim*[np.nan])

    def insert_virtual_obs(self, detections_dict, curr_state):
        #insert virtual observation using
        return

    def insert_real_obs(self, detections_dict):
        return

    def add_pose_node(self):
        #adds a new pose after a given space/time
        return

    def get_nearest_neighbor(self):
        #finds the closest spatial pose graph node
        return

    def compute_loop_closure_constraint(self):
        #computes new transform between two nodes
        return

    def compute_node_similarity(self, node_1, node_2):
        #for vision based BoW or other feature matching technique
        return

    def remove_pose_node(self, pose_node):
        #for pruning the pose graph
        return

    def publish_sampled_odom_poses(self):
        now = rospy.Time(0)
        try:
            (trans, rot) = self.tf.lookupTransform("odom", "base_link", now)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        pos = np.asarray(trans)
        self.create_marker(pos, self.marker_id % 500, "odom", self.marker_dict, [0, 1, 0, .75])
        self.marker_id += 1
        self.marker_pub.publish(self.marker_dict.values())

    def publish_detections(self):
        self.data_mutex.acquire()
        for det_id in self.detections_dict:
            pos = self.detections_dict[det_id][0]
            frame_id = self.detections_dict[det_id][1]
            self.create_marker(pos, det_id, "odom", self.detection_marker_dict, self.color_cycle[self.flip % 2], True)
        self.data_mutex.release()
        self.flip += 1
        if len(self.detection_marker_dict.keys()) > 0:
            self.detection_pub.publish(self.detection_marker_dict.values())
        else:
            self.detection_pub.publish([self.clear_mark])

    def create_marker(self, pos, id, frame, dict, color, arrow=False):
        m = Marker()
        m.header.frame_id = frame
        m.id = id
        m.type = 2

        if arrow:
            m.type = 0

        if arrow:
            (trans, rot) = self.tf.lookupTransform("odom", "base_link", rospy.Time(0))
            m.points = [array_to_point(trans), array_to_point(pos)]
        else:
            m.pose.position.x = pos[0]
            m.pose.position.y = pos[1]
            m.pose.position.z = pos[2]

        m.color.r = color[0]
        m.color.g = color[1]
        m.color.b = color[2]
        m.color.a = .50
        m.scale.x = .1
        m.scale.y = .1
        m.scale.z = .1
        dict[id] = m

    def create_system(self):
        #create structures needed to solve states
        return

    def solve(self):
        #update internal node states
        return

    # pose_graph spin loop optimizes at a set frequency
    def spin(self, event):
        self.publish_sampled_odom_poses()
        self.publish_detections()



def main():
    rospy.init_node('pose_graph', anonymous=True)
    pg = Pose_Graph()
    rospy.spin()
