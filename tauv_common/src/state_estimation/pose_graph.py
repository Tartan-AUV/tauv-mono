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
        self.marker_dict = {}
        self.marker_id = 0
        self.new_measurement = False
        self.measurement_server = rospy.Service("pose_graph/register_measurement", RegisterMeasurement, \
                                              self.register_measurement)
        self.marker_callback = rospy.Timer(rospy.Duration(.1), self.publish_sampled_odom_poses)

    def array_to_point(self, arr):
        p = Point()
        p.x = arr[0]
        p.y = arr[1]
        p.z = arr[2]
        return p

    def point_to_array(self, point):
        return np.asarray([point.x, point.y, point.z])

    def transform_meas_to_frame(self, measurement, child_frame, world_frame, time):
        try:
            (trans, rot) = self.tf.lookupTransform(world_frame, child_frame, time)
            tf = R.from_quat(np.asarray(rot))
            detection_pos = tf.apply(measurement) + np.asarray(trans)
            return detection_pos
        except:
            return np.array([np.nan])

    def register_measurement(self, req):
        print("In server")
        meas = req.pg_meas
        pos = self.point_to_array(meas.position)
        id = meas.landmark_id
        pos = np.asarray(self.transform_meas_to_frame(pos, meas.header.frame_id, "base_link", meas.header.stamp))
        print(pos, id)
        return True


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

    def publish_sampled_odom_poses(self, event):
        now = rospy.Time(0)
        try:
            (trans, rot) = self.tf.lookupTransform("odom", "base_link", now)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        pos = np.asarray(trans)
        self.create_marker(pos, self.marker_id % 50)
        self.marker_id += 1
        self.marker_pub.publish(self.marker_dict.values())

    def create_marker(self, pos, id):
        m = Marker()
        m.header.frame_id = "odom"
        m.id = id
        m.type = 2
        m.pose.position.x = pos[0]
        m.pose.position.y = pos[1]
        m.pose.position.z = pos[2]
        m.color.r = 1.0
        m.color.a = 1.0
        m.scale.x = .1
        m.scale.y = .1
        m.scale.z = .1
        self.marker_dict[id] = m

    def create_system(self):
        #create structures needed to solve states
        return

    def solve(self):
        #update internal node states
        return

    def run_grapher(self):
        return


def main():
    rospy.init_node('pose_graph', anonymous=True)
    pg = Pose_Graph()
    rospy.spin()
