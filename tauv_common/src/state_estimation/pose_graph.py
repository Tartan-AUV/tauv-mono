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
        self.factor_relations




class Pose_Graph():
    def __init__(self, freq):
        self.front_image_feed = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_image", Image, self.left_callback)
        self.rate = rospy.Rate(freq)
        self.nodes = {"landmarks":dict(), "poses":dict()}


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

    def create_system(self):
        #create structures needed to solve states

    def solve(self):
        #update internal node states

    def run_grapher(self):
