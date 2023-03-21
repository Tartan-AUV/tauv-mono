#!/usr/bin/env python3

# Author: Gleb Ryabtsev, 2023

import rospy
import cv2
import cv_bridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from tauv_util.parms import Parms

from vision.detectors.gate_detector_new import GateDetector

class GateDetectorNode:
    def __init__(self):
        parameters = Parms(rospy.get_param("~gate_detector_parameters"))
        self._detector = GateDetector(parameters)

        self._rgb_sub = message_filters.Subscriber('color', Image)
        self._depth_sub = message_filters.Subscriber('depth_map', Image)
        self._camera_info_sub = message_filters.Subscriber('camera_info',
                                                           CameraInfo)

        self._time_synchronizer = message_filters.TimeSynchronizer(
            [self._rgb_sub,
            self._depth_sub,
               self._camera_info_sub], 10)
        self._time_synchronizer.registerCallback(self.callback)

        self._detection_pub = rospy.Publisher('gate_detections', PoseArray,
                                              queue_size=10)
        self._frame_id = rospy.get_param("~frame_id")

        self._bridge = cv_bridge.CvBridge()

    def callback(self, rgb: Image, depth: Image, camera_info: CameraInfo):
        # Update camera matrix
        self._detector.set_camera_matrix(camera_info.K)

        # Convert colors
        cv_rgb = self._bridge.imgmsg_to_cv2(rgb, desired_encoding="passthrough")
        cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        cv_depth = self._bridge.imgmsg_to_cv2(depth,
                                              desired_encoding="passthrough")

        # Get detections
        detections = self._detector.detect(cv_rgb, cv_depth)

        # Convert to PoseArray and publish
        pose_array = PoseArray()
        pose_array.header.stamp = rgb.header.stamp
        pose_array.header.frame_id = self._frame_id
        for pos in detections:
            pose = Pose()
            pose.position = Point(pos.x, pos.y, pos.z)
            pose.orientation = Quaternion(pos.q0, pos.q1, pos.q2, pos.q3)
            pose_array.poses.append(pose)

        self._detection_pub.publish(pose_array)


def main():
    rospy.init_node('gate_detector', anonymous=True)
    g = GateDetectorNode()
    rospy.spin()

main()
