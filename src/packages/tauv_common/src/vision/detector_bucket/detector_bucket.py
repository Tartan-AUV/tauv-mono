import rospy
import numpy as np
from cv_bridge import CvBridge
from typing import Dict

from tauv_msgs.msg import BucketDetection
from .detector_kf import DetectorKF

class DetectorBucket():
    def __init__(self):
        self._load_config()

        self._detection_sub: rospy.Subscriber = rospy.Subscriber('detection', BucketDetection, self._handle_detection)

        self._kfs: Dict[str, DetectorKF] = dict(map(lambda tag: (tag, DetectorKF(tag)), self._tags))

    def start(self):
        rospy.Timer(self._dt, self._update)
        rospy.spin()

    def _update(self, timer_event):
        pass

    def _handle_detection(self, msg: BucketDetection):
        self._kfs[msg.tag].handle_pose(msg.pose)
        pose = self._kfs[msg.tag].get_pose()

    def _load_config(self):
        self._dt: rospy.Duration = rospy.Duration.from_sec(1.0 / rospy.get_param('~frequency'))
        self._tags: [str] = rospy.get_param('~tags')


def main():
    rospy.init_node('detector_bucket')
    n = DetectorBucket()
    n.start()
