#!/usr/bin/env python3

import rospy
from tauv_msgs.msg import Pose, BucketDetection, BucketList, RegisterObjectDetections,BoundingBoxes, BoundingBox
from std_msgs.msg import Header
import numpy as numpy
import cv2
from cv_bridge import CvBridge
#from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from sensor_msgs.msg import Image, CameraInfo
from vision.depth_estimation.depth_estimation import DepthEstimator
from geometry_msgs.msg import Point
from vision.detector_bucket.detector_bucket import Detector_Bucket
import math
from tauv_util import transforms
from tauv_util import types
#from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger

FT = 0.3048
IN = FT / 12

FORCE_DEPTH_BOTTOM = 4 * FT + 3 * IN

class LogDetections():
    def __init__(self):
        self.cur_position = (0,0,0)
        self.cur_orientation = (0,0,0)

        self.front_camera = "zedm_A_left_camera_optical_frame"

        rospy.init_node('image_detector', anonymous = True)

        self.front_camera_info = None
        self.front_depth_image = None
        self.front_image_ready = False

        self.bottom_camera_info = None
        self.bottom_depth_image = None
        self.bottom_image_ready = False

        self.bounding_boxes = BoundingBoxes()
        self.cv_bridge = CvBridge()

        rospy.wait_for_message("/darknet_ros/bounding_boxes", BoundingBoxes)
        
        rospy.Subscriber("/zedm_A/zed_node_A/depth/depth_registered",Image,self.front_depth_callback)
        rospy.Subscriber("/zedm_A/zed_node_A/left/camera_info",CameraInfo, self.front_camera_info_callback)

        rospy.Subscriber("/zedm_B/zed_node_B/depth/depth_registered",Image,self.bottom_depth_callback)
        rospy.Subscriber("/zedm_B/zed_node_B/left/camera_info",CameraInfo, self.bottom_camera_info_callback)

        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)

        self.detector = rospy.Publisher("register_object_detection", RegisterObjectDetections,
                                              queue_size=10)

        rospy.Subscriber("gnc/pose", Pose, self.update_position)

    def update_position(self,data):
        self.cur_position = (data.position.x, data.position.y, data.position.z)
        self.cur_orientation = (data.orientation.x, data.orientation.y, data.orientation.z)

    def front_camera_info_callback(self, msg):
        self.front_camera_info = msg

    def front_depth_callback(self, msg):
        self.front_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.front_image_ready = True

    def bottom_depth_callback(self, msg):
        self.bottom_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.bottom_image_ready = True

    def bottom_camera_info_callback(self, msg):
        self.bottom_camera_info = msg
        
    def calc_front_pos(self, relative, cur_orientation, cur_position):
        #print("FRONT")
        roll = cur_orientation [0]
        pitch = cur_orientation [1]
        yaw = cur_orientation [2]

        relative = numpy.array([relative[2] + 0.465, relative[0], relative[1]])

        rpy = numpy.array([[math.cos(pitch)*math.cos(yaw), -math.cos(roll)*math.sin(yaw)+math.cos(yaw)*math.sin(pitch)*math.sin(roll), math.sin(roll)*math.sin(yaw)+math.cos(roll)*math.sin(pitch)*math.cos(yaw)],
        [math.cos(pitch)*math.sin(yaw), math.cos(roll)*math.cos(yaw)+math.sin(roll)*math.sin(pitch)*math.sin(yaw), -math.sin(roll)*math.cos(yaw)+math.cos(roll)*math.sin(pitch)*math.sin(yaw)],
        [-math.sin(pitch), math.sin(roll)*math.cos(pitch), math.cos(roll)*math.cos(pitch)]])

        return array_to_point(cur_position+numpy.matmul(rpy, relative))

    def calc_bottom_pos(self, relative, cur_orientation, cur_position):
        #print("BOTTOM")
        roll = cur_orientation [0]
        pitch = cur_orientation [1]
        yaw = cur_orientation [2]

        # scale relative vector to force depth
        fd = FORCE_DEPTH_BOTTOM
        real_z_dist = fd - self.cur_position[2]
        thought_z_dist = relative[2]

        real_x = relative[0] * real_z_dist / thought_z_dist
        real_y = relative[1] * real_z_dist / thought_z_dist


        relative = numpy.array([-real_x+0.3302, -real_y, fd])

        rpy = numpy.array([[math.cos(pitch)*math.cos(yaw), -math.cos(roll)*math.sin(yaw)+math.cos(yaw)*math.sin(pitch)*math.sin(roll), math.sin(roll)*math.sin(yaw)+math.cos(roll)*math.sin(pitch)*math.cos(yaw)],
        [math.cos(pitch)*math.sin(yaw), math.cos(roll)*math.cos(yaw)+math.sin(roll)*math.sin(pitch)*math.sin(yaw), -math.sin(roll)*math.cos(yaw)+math.cos(roll)*math.sin(pitch)*math.sin(yaw)],
        [-math.sin(pitch), math.sin(roll)*math.cos(pitch), math.cos(roll)*math.cos(pitch)]])

        return array_to_point(cur_position+numpy.matmul(rpy, relative))


    def invalid_pos(self, objdet):
        if(numpy.isnan(objdet.position.x) or numpy.isnan(objdet.position.y) or numpy.isnan(objdet.position.z)):
            return True
        
        if(objdet.position.z<0):
            return True

        return False


    def bbox_callback(self, bboxes):
        rospy.loginfo("BBOX ALERT")

        sub_or = self.cur_orientation
        sub_pos = self.cur_position
        front = bboxes.image_header.frame_id==self.front_camera #is front camera?

        if((front and (not self.front_image_ready)) or ((not front) and (not self.bottom_image_ready))):
            return

        objects = RegisterObjectDetections()
        objects.objdets = list()

        #rospy.loginfo(f"box : {bboxes.bounding_boxes}")

        for bbox in bboxes.bounding_boxes:
            objdet = BucketDetection()

            objdet.tag = bbox.Class

            if objdet.tag == "badge" and bbox.probability < 0.3:
                continue

            if(len(objects.objdets) > 2):
                continue

            relative_pos = DepthEstimator.estimate_absolute_depth(self.front_depth_image, bbox, self.front_camera_info)

            if(relative_pos == numpy.nan):
                continue

            objdet.position = (numpy.nan, numpy.nan, numpy.nan)
            if(front):
                objdet.position = self.calc_front_pos(relative_pos, sub_or, sub_pos)
            else:
                objdet.position = self.calc_bottom_pos(relative_pos, sub_or, sub_pos)


            if(self.invalid_pos(objdet)):
                continue

            #rospy.loginfo(f"CALCULATE pos: {objdet.position}\n")
        
            objects.objdets.append(objdet)

        objects.detector_tag = "camera"


        self.detector.publish(objects)


def array_to_point(arr):
    p = Point()
    p.x = arr[0]
    p.y = arr[1]
    p.z = arr[2]
    return p

def main():
    s = LogDetections()
    rospy.spin()