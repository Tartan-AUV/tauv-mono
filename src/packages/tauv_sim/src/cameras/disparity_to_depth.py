#!/usr/bin/env python3

import rospy
import cv2 as cv
from sensor_msgs.msg import Image, CameraInfo
from tauv_msgs.srv import GetCameraInfo, GetCameraInfoResponse
from stereo_msgs.msg import DisparityImage
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

NODENAME = "oakd_ros"
QUEUE_SIZE = 100
FPS = 30


class DisparityToDepthNode:
    def __init__(self):
        rospy.init_node(NODENAME, anonymous=True)
        self.__bridge = CvBridge()
        self.__sub = rospy.Subscriber("disparity", DisparityImage, callback=self.__cb, queue_size=10)
        self.__pub = rospy.Publisher("depth_map", Image, queue_size=10)

    def __cb(self, msg: DisparityImage):
        disp_img = self.__bridge.imgmsg_to_cv2(msg.image)
        depth_img = np.where(disp_img > 0, (msg.T * msg.f) / disp_img, 0)
        depth_msg = Image()
        depth_msg.header = msg.header
        depth_msg = self.__bridge.cv2_to_imgmsg(depth_img, encoding="passthrough", header=depth_msg.header)
        self.__pub.publish(depth_msg)


def main():
    DisparityToDepthNode()
    rospy.spin()


main()