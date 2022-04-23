import rospy
import numpy as np

from geometry_msgs.msg import Pose
from tauv_util.types import tl
from tauv_util.transforms import quat_to_rpy


class DetectorKF:
    NUM_FIELDS = 6

    def __init__(self, tag):
        self._F: np.array = np.identity(DetectorKF.NUM_FIELDS, float)
        self._H: np.array = np.identity(DetectorKF.NUM_FIELDS, float)
        self._R: np.array = 1.0e6 * np.identity(DetectorKF.NUM_FIELDS, float)
        self._Q: np.array = 1.0e-3 * np.identity(DetectorKF.NUM_FIELDS, float)
        self._x: np.array = np.zeros(DetectorKF.NUM_FIELDS, float)
        self._P: np.array = 1.0e6 * np.identity(DetectorKF.NUM_FIELDS, float)

    def handle_pose(self, measurement: Pose):
        m: np.array = np.concatenate((
            tl(measurement.position),
            quat_to_rpy(measurement.orientation),
        ))

        self._predict()
        self._update(m)

    def _predict(self):
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ np.transpose(self._F) + self._Q

    def _update(self, measurement: np.array):
        y: np.array = measurement - self._H * self._x
        S: np.array = (self._H @ self._P) @ np.transpose(self._H) + self._R
        K: np.array = (self._P @ np.transpose(self._H)) @ np.linalg.pinv(S)

        self._x = self._x + K @ y
        self._P = self._P - K @ self._H @ self._P

