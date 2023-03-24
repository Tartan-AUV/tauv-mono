import rospy
import threading
import numpy as np
import tf2_ros as tf2
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from tauv_msgs.msg import FeatureDetections
import cv2
from math import cos, sin

class ShapeDetector:

    def __init__(self):
        self._lock: threading.Lock = threading.Lock()
        self._lock.acquire()

        self._load_config()

        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer)

        self._cv_bridge: CvBridge = CvBridge()

        self._camera_info: CameraInfo = rospy.wait_for_message(f'vehicle/{self._frame_id}/color/camera_info', CameraInfo, 60)
        self._intrinsics: np.array = np.array(self._camera_info.K).reshape((3, 3))
        self._distortion: np.array = np.array(self._camera_info.D)

        self._img_sub: rospy.Subscriber = rospy.Subscriber(f'vehicle/{self._frame_id}/color/image_raw', Image, self._handle_img)
        self._detections_pub: rospy.Publisher = rospy.Publisher('global_map/feature_detections', FeatureDetections, queue_size=10)

        self._lock.release()

    def start(self):
        rospy.spin()

    def _handle_img(self, msg):
        self._lock.acquire()

        img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = img_hsv.shape

        img_thresh = np.zeros((h, w), dtype=np.uint8)
        for hsv_range in self._hsv_ranges:
            low = np.array([hsv_range[0], hsv_range[2], hsv_range[4]])
            high = np.array([hsv_range[1], hsv_range[3], hsv_range[5]])
            mask = cv2.inRange(img_hsv, low, high)
            img_thresh = img_thresh | mask

        cv2.imshow('img_thresh', 255 * img_thresh)

        img_blur = cv2.GaussianBlur(img_thresh, (5, 5), 0)

        cv2.imshow('img_blur', img_blur)

        contours, _ = cv2.findContours(img_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours = [contour for contour in contours if cv2.contourArea(contour) >= self._contour_area_threshold]

        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)

        for contour in contours:
            ellipse = cv2.fitEllipse(contour)

            cv2.ellipse(img_contours, ellipse, (255, 0, 0), 1)

            flat_contour = contour.reshape((contour.shape[0], contour.shape[2]))

            max_idx = np.argmax(flat_contour, axis=0)
            min_idx = np.argmin(flat_contour, axis=0)

            max_x, max_y = flat_contour[max_idx]
            min_x, min_y = flat_contour[min_idx]

            points_image = np.array([
                min_y,
                max_x,
                max_y,
                min_x,
            ]).astype(np.float64)

            c_w = self._circle_width
            points_world = np.array([
                [0, c_w / 2, 0],
                [c_w / 2, 0, 0],
                [0, -c_w / 2, 0],
                [-c_w / 2, 0, 0],
            ], dtype=np.float64)

            print(points_world)
            print(self._intrinsics)
            print(self._distortion)
            res, rvec, tvec = cv2.solvePnP(points_world, points_image, self._intrinsics, self._distortion, flags=0)

            axes_world = np.array([
                [0, 0, 0],
                [0.1, 0, 0],
                [0, 0.1, 0],
                [0, 0, 0.1],
            ], dtype="double")
            axes_image, _ = cv2.projectPoints(axes_world, rvec, tvec, self._intrinsics, self._distortion)
            axes_image = np.array(axes_image).astype(int)

            cv2.line(img_contours, tuple(axes_image[0,0,:]), tuple(axes_image[1,0,:]), color=(0, 0, 255), thickness=3)
            cv2.line(img_contours, tuple(axes_image[0,0,:]), tuple(axes_image[2,0,:]), color=(0, 255, 0), thickness=3)
            cv2.line(img_contours, tuple(axes_image[0,0,:]), tuple(axes_image[3,0,:]), color=(255, 0, 0), thickness=3)

        cv2.imshow('img_contours', img_contours)

        cv2.waitKey(1000)

        self._lock.release()

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame_id = rospy.get_param('~frame_id')
        self._hsv_ranges = rospy.get_param('~hsv_ranges')
        self._contour_area_threshold = rospy.get_param('~contour_area_threshold')
        self._circle_width = rospy.get_param('~circle_width')


def main():
    rospy.init_node('shape_detector')
    n = ShapeDetector()
    n.start()