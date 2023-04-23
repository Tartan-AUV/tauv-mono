import rospy
import numpy as np
from tauv_msgs.msg import XsensImuData as ImuMsg, FluidDepth as DepthMsg, NavigationState


class AcousticStateEstimation:

    def __init__(self):
        self._dt: float = 0.02
        self._navigation_state_pub: rospy.Publisher = rospy.Publisher('gnc/navigation_state', NavigationState, queue_size=10)

        self._imu_sub: rospy.Subscriber = rospy.Subscriber('vehicle/xsens_imu/data', ImuMsg, self._handle_imu)
        self._depth_sub: rospy.Subscriber = rospy.Subscriber('vehicle/arduino/depth', DepthMsg, self._handle_depth)

    def _update(self, timer_event):
        pass
        # Publish orientation, angular rates, and depth

    def start(self):
        rospy.Timer(self._dt, self._update)
        rospy.spin()

    def _handle_imu(self, msg: ImuMsg):
        pass

    def _handle_depth(self, mgs: DepthMsg):
        pass


def main():
    rospy.init_node('acoustic_state_estimation')
    n = AcousticStateEstimation()
    n.start()

