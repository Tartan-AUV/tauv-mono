#!/usr/bin/env python3

import rospy
import depthai
import cv2 as cv
from sensor_msgs.msg import Image, CameraInfo
from tauv_msgs.srv import GetCameraInfo, GetCameraInfoResponse
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

NODENAME = "oakd_ros"
QUEUE_SIZE = 100
FPS = 30
# node to create oakd ros nodes from oakd api
# publishes depth map and color image

class OAKDNode:
    def __init__(self):
        rospy.init_node(NODENAME, anonymous = True)

        self.pipeline = depthai.Pipeline()

        self.front_cam_rgb = self.pipeline.create(depthai.node.ColorCamera)
        self.front_stereo = self.pipeline.create(depthai.node.StereoDepth)
        self.front_left = self.pipeline.create(depthai.node.MonoCamera)
        self.front_right = self.pipeline.create(depthai.node.MonoCamera)

        self.front_left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        self.front_right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

        self.front_xout_color = self.pipeline.create(depthai.node.XLinkOut)
        self.front_xout_depth = self.pipeline.create(depthai.node.XLinkOut)

        self.front_xout_color.setStreamName("rgb")
        self.front_xout_depth.setStreamName("depth")

        self.front_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
        self.front_left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        self.front_left.setFps(FPS)

        self.front_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
        self.front_right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)
        self.front_right.setFps(FPS)

        self.front_cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
        self.front_cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.front_cam_rgb.setInterleaved(False)
        self.front_cam_rgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
        self.front_cam_rgb.setFps(FPS)
        
        self.front_stereo.setLeftRightCheck(False)
        self.front_stereo.setExtendedDisparity(False)
        self.front_stereo.setSubpixel(False)
        self.front_stereo.setDepthAlign(depthai.CameraBoardSocket.RGB)
        self.front_stereo.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        self.front_stereo.initialConfig.setMedianFilter(depthai.MedianFilter.KERNEL_7x7)

        self.front_left.out.link(self.front_stereo.left)
        self.front_right.out.link(self.front_stereo.right)
        self.front_stereo.depth.link(self.front_xout_depth.input)
        self.front_cam_rgb.video.link(self.front_xout_color.input)

        self.front_device = depthai.Device(self.pipeline)
        
        self.front_calibData = self.front_device.readCalibration()
        self.front_camera_info = CameraInfo()
        self.front_camera_info.K = np.ndarray.flatten(np.array(self.front_calibData.getCameraIntrinsics(depthai.CameraBoardSocket.RGB)))
        self.front_camera_info.distortion_model = "rational_polynomial"
        self.front_camera_info.D = np.array(self.front_calibData.getDistortionCoefficients(depthai.CameraBoardSocket.RGB))

        self.bridge = CvBridge()

        #estimate of ros system time offset compared to depthai clock
        depthai_time = depthai.Clock.now()
        self.time_offset = rospy.Time.now() - rospy.Time.from_sec(depthai_time.total_seconds())
        print(self.time_offset)

        self.front_depthPub = rospy.Publisher("/oakd/oakd_front/depth_map", Image, queue_size=QUEUE_SIZE)
        self.front_colorPub = rospy.Publisher("/oakd/oakd_front/color_image", Image, queue_size=QUEUE_SIZE)
        self.cameraInfoService = rospy.Service("/oakd/camera_info", GetCameraInfo, self.camera_info_service)

        self.spin()

    def camera_info_service(self, req):
        if(req.camera_name=="oakd_front"):
            resp = GetCameraInfoResponse()
            resp.camera_info = self.front_camera_info
            return resp

        return None

    def spin(self):
        qRgb = self.front_device.getOutputQueue(name="rgb", maxSize=QUEUE_SIZE, blocking=False)
        qDepth = self.front_device.getOutputQueue(name="depth", maxSize=QUEUE_SIZE, blocking=False)

        while True:
            rgb = qRgb.tryGet()
            depth = qDepth.tryGet()

            if rgb is not None:
                try:
                    time_diff = rgb.getTimestamp()

                    img = self.bridge.cv2_to_imgmsg(rgb.getCvFrame(), encoding='bgr8')
                    img.header.frame_id = "oakd_front"
                    img.header.seq = rgb.getSequenceNum()
                    img.header.stamp = self.time_offset + rospy.Time.from_sec(rgb.getTimestamp().total_seconds())
                    self.front_colorPub.publish(img)
                except CvBridgeError as e:
                    rospy.loginfo("OAKD frame error")

            if depth is not None:
                try:
                    img = self.bridge.cv2_to_imgmsg(depth.getCvFrame(), encoding='mono16')
                    img.header.frame_id = "oakd_front"
                    img.header.seq = depth.getSequenceNum()
                    img.header.stamp = self.time_offset + rospy.Time.from_sec(depth.getTimestamp().total_seconds())
                    self.front_depthPub.publish(img)
                except CvBridgeError as e:
                    rospy.loginfo("OAKD frame error")

def main():
    OAKDNode()
    rospy.spin()

main()