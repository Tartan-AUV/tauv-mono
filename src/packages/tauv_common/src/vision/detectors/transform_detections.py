#!/usr/bin/env python3

import rospy
from tauv_msgs.msg import Pose, FeatureDetection, FeatureDetections
from tauv_msgs.srv import GetCameraInfo
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from sensor_msgs.msg import Image
from vision.depth_estimation.depth_estimation import DepthEstimator
from geometry_msgs.msg import Point, PointStamped, Point, Vector3
import math
from tauv_util import transforms
from tauv_util import types
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger
import tf

class TransformDetections():
    def __init__(self):
        self.NODE_NAME = "transform_detections"
        self.NODE_NAME_FMT = "[{}]".format(self.NODE_NAME)

        rospy.init_node(self.NODE_NAME, anonymous = True)

        self.tf_listener = tf.TransformListener()

        self.front_camera_frame_id = "oakd_front"
        self.bottom_camera_frame_id = "oakd_bottom"

        self.frames_by_frame_id = {
            self.front_camera_frame_id: dict()
        #    ,self.bottom_camera_frame_id: dict()
        }

        self.cv_bridge = CvBridge()

        rospy.wait_for_service('/oakd/camera_info')
        self.camera_info = rospy.ServiceProxy('/oakd/camera_info', GetCameraInfo)
        self.front_camera_info = None
        try:
            resp = self.camera_info("oakd_front")
            self.front_camera_info = resp.camera_info
        except rospy.ServiceException as exc:
            rospy.logerr("Camera info failed")
        
        # initialize subscribers for front camera depthmaps and camera info
        rospy.Subscriber("/oakd/oakd_front/depth_map", Image, self.add_depth_frame)

        # initialize subscriber for darknet NN bounding boxes
        self.bounding_boxes = rospy.wait_for_message("/darknet_ros/bounding_boxes", BoundingBoxes)
        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.add_bbox_frame)

        self.detector = rospy.Publisher("/global_map/transform_detections", FeatureDetections,
                                        queue_size=10)


    def start(self):
        rospy.spin()


    def add_depth_frame(self, frame):
        seq = frame.header.seq
        frame_id = frame.header.frame_id

        # adds current camera data to dictionary
        if not seq in self.frames_by_frame_id[frame_id]:
            self.frames_by_frame_id[frame_id][seq] = {}

        self.frames_by_frame_id[frame_id][seq]['depth'] = self.cv_bridge.imgmsg_to_cv2(frame, desired_encoding='mono16')
        self.frames_by_frame_id[frame_id][seq]['time'] = frame.header.stamp

        # check if dictionary frame seq has correspnding bboxes entry
        if('bboxes' in self.frames_by_frame_id[frame_id][seq]):
            self.transform_matched_frames(frame_id,seq)


    def add_bbox_frame(self, frame):
        seq = frame.image_header.seq
        frame_id = frame.image_header.frame_id

        # adds bbox data to dictionary
        if not seq in self.frames_by_frame_id[frame_id]:
            self.frames_by_frame_id[frame_id][seq] = {}

        self.frames_by_frame_id[frame_id][seq]['bboxes'] = frame.bounding_boxes

        # check if dictionary frame seq has correspnding bboxes entry
        if('depth' in self.frames_by_frame_id[frame_id][seq]):
            self.transform_matched_frames(frame_id,seq)
        

    def transform_matched_frames(self, frame_id, seq):
        bboxes = self.frames_by_frame_id[frame_id][seq]['bboxes']
        depth = self.frames_by_frame_id[frame_id][seq]['depth']
        time = self.frames_by_frame_id[frame_id][seq]['time']

        # new list of all transformed detections in current frame
        objects = FeatureDetections()
        objects.detections = list()
        objects.header = Header()
        objects.header.frame_id = frame_id
        objects.header.stamp = time
        objects.header.seq = seq

        for bbox in bboxes:
            objdet = FeatureDetection()
            objdet.tag = bbox.Class

            # calculate depth of the object in a relative coordinate frame, returned as an (x, y, z) in NED frame
            relative_pos = DepthEstimator.estimate_absolute_depth(depth,
                                                                  bbox,
                                                                  self.front_camera_info)

            if relative_pos == np.nan: # invalid depth estimate
                continue

            # transform point from sensor coordinate frame to world coordinate frame
            # (sensor_trans, sensor_rot) = self.tf_listener.lookupTransform(
            #     "odom_ned",
            #     frame_id,
            #     time
            # )

            # wrapped relative position
            relative_trans = PointStamped()
            relative_trans.header.stamp = time
            relative_trans.header.frame_id = frame_id

            relative_trans.point.x = relative_pos[0]
            relative_trans.point.y = relative_pos[1]
            relative_trans.point.z = relative_pos[2]

            global_point = self.tf_listener.transformPoint(
                "odom_ned",
                relative_trans
            )

            # trans = [relative_pos[0], relative_pos[1], relative_pos[2], 1]
            # trans_position = np.matmul(self.tf_listener.fromTranslationRotation(trans,rot), trans)
            objdet.position = Vector3(global_point.point.x, global_point.point.y, global_point.point.z)
            objects.detections.append(objdet)

        #delete used frame from dict
        del self.frames_by_frame_id[frame_id][seq]

        self.detector.publish(objects)

def main():
    s = TransformDetections()
    s.start()