import cv2
import cv_bridge
import numpy as np
import rospy
from sensor_msgs.msg import Image


def generate_mask_in_hue_range(image_hsv, lower_hue, upper_hue):
    if upper_hue < lower_hue:
        # lower mask (0 - upper_hue)
        lower = np.array([0, 50, 50])
        upper = np.array([upper_hue, 255, 255])
        mask0 = cv2.inRange(image_hsv, lower, upper)

        # upper mask (lower_hue - 180)
        lower = np.array([lower_hue, 50, 50])
        upper = np.array([180, 255, 255])
        mask1 = cv2.inRange(image_hsv, lower, upper)
        return mask0 + mask1
    else:
        lower = np.array([lower_hue, 50, 50])
        upper = np.array([upper_hue, 255, 255])
        return cv2.inRange(image_hsv, lower, upper)


class HSVFilter:
    def __init__(self):
        # HSV format, hue channel range is [0, 180], setting higher < lower indicates wrapping around 180 -> 0
        self._lower_hue: int = 170  # default red
        self._upper_hue: int = 10  # default red
        self._cv_bridge: cv_bridge.CvBridge = cv_bridge.CvBridge()

        self._sub: rospy.Subscriber = (
            rospy.Subscriber("/kf/vehicle/oakd_bottom/color/image_raw", Image, self._handle_image)
        )
        self._pub_mask: rospy.Publisher = rospy.Publisher("hsv_filter/image_mask", Image, queue_size=10)
        self._pub_threshold: rospy.Publisher = rospy.Publisher("hsv_filter/image_threshold", Image, queue_size=10)
        # self._srv: rospy.Service = rospy.Service("hsv_filter/new_hue_range", Trigger, self._handle_hue_range)

    def _handle_image(self, msg):
        image_raw = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        image_hsv = cv2.cvtColor(image_raw, cv2.COLOR_BGR2HSV)
        image_mask = generate_mask_in_hue_range(image_hsv, self._lower_hue, self._upper_hue)
        msg_mask = self._cv_bridge.cv2_to_imgmsg(image_mask, encoding='brg8')
        self._pub_mask.publish(msg_mask)

        image_threshold = image_raw.copy()
        image_threshold[np.where(image_mask == 0)] = 0
        msg_threshold = self._cv_bridge.cv2_to_imgmsg(image_threshold, encoding='brg8')
        self._pub_threshold.publish(msg_threshold)


def main():
    hsv_filter = HSVFilter()
    rospy.spin()
