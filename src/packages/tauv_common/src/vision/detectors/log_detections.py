#!/usr/bin/env python3

import rospy
from tauv_msgs.msg import Pose, BucketDetection, Header, BucketList
from std_msgs.msg import Header
import numpy as numpy
import cv2
from darknet_ros.msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image, CameraInfo
from depth_estimation import DepthEstimator
from detector_bucket import Detector_Bucket
from tauv_msgs.srv import RegisterObjectDetections
import math

class LogDetections():
    def __init__(self):
        self.cur_position = (0,0,0)
        self.cur_orientation = (0,0,0)

        self.new_image = False

        self.depth_estimator = DepthEstimator()

        self.depth_camera_info = CameraInfo()
        self.bounding_boxes = BoundingBoxes()
        
        rospy.init_node('image_detector', anonymous = True)
        
        self.depth_image_streamer = rospy.Subscriber("zedm_A/zed_node_A/depth/depth_registered",Image,self.depth_callback)
        self.depth_camera_info = rospy.Subscriber("/zedm_A/zed_node_A/left/camera_info",CameraInfo, self.camera_info_callback)
        self.bounding_boxes = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)
        self.detector = rospy.Publisher("register_object_detection", RegisterObjectDetections, \
                                              queue_size=10)

        rospy.Subscriber("gnc/pose", Pose, self.update_position)
        

    def update_position(self,data):
        self.cur_position = (data.position.x, data.position.y, data.position.z)
        self.cur_orientation = (data.orientation.x, data.orientation.y, data.orientation.z)

    def camera_info_callback(self, msg):
        self.depth_camera_info = msg

    def depth_callback(self, msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")

    def calc_pos(self, relative):
        roll = self.cur_orientation.x
        pitch = self.cur_orientation.y
        yaw = self.cur_orientation.z

        (xr,yr,zr) = (relative.x*math.cos(roll) - relative.y*math.sin(roll), relative.x*math.sin(roll)+relative.y*math.cos(roll), relative.z)
        (xrp, yrp, zrp) = (xr, yr*math.cos(pitch)+zr*math.sin(pitch),zr*math.cos(pitch)-yr*math.sin(pitch))
        (xrpy, yrpy, zrpy) = (xrp*math.cos(yaw)-zrp*math.sin(yaw), yrp, xrp*math.sin(yaw)+zrp*math.cos(yaw))

        return (self.cur_position.x + zrpy, self.cur_position.y + xrpy, self.cur_position.z + yrpy)

    def bbox_callback(self, bboxes):
        objects = RegisterObjectDetections()
        objects.objdets = list()

        for bbox in bboxes:
            objdet = BucketDetection()
            objdet.tag = bbox.Class

            relative_pos = self.depth_estimator(self.depth_image, bbox, self.depth_camera_info)

            objdet.position = self.calc_pos(relative_pos)

            objects.objdets.append(objdet)

        objects.detector_tag = "camera"

        self.detector.publish(objects)


def main():
    s = LogDetections()
    rospy.spin()