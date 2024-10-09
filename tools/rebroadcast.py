import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class Rebroadcaster:

    def __init__(self):
        self._cv_bridge = CvBridge()

        self._img_sub = rospy.Subscriber('/kf/vehicle/oakd_front/color/image_raw', Image, self._handle_image)
        self._img_pub = rospy.Publisher('/kf/darknet_ros/image', Image, queue_size=10)

        # img = cv2.imread("/home/tauv/buoy-1_0_buoy_earth_2_buoy_earth_1_buoy_abydos_2_buoy_abydos_1_00180.png")
        # img_rgb = img[:, :, ::-1]
        # img_msg = self._cv_bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
        # while True:
        #     self._img_pub.publish(img_msg)
        #     time.sleep(1.0)

    def _handle_image(self, msg: Image):
        img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = img[:, :, ::-1]
        out_msg = self._cv_bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self._img_pub.publish(out_msg)


def main():
    rospy.init_node('rebroadcaster')
    n = Rebroadcaster()
    print('starting')
    rospy.spin()
    print('stopped')

if __name__ == "__main__":
    main()