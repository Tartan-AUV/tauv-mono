import rospy
import threading
import numpy as np
import tf2_ros as tf2
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from tauv_msgs.msg import FeatureDetections, FeatureDetection
from tauv_util.transforms import tf2_transform_to_homogeneous, tf2_transform_to_quat, multiply_quat, quat_to_rpy
from geometry_msgs.msg import Point, Quaternion
import cv2
from math import cos, sin, pi, atan2
from scipy.spatial.transform import Rotation

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
        self._contours_img_pub: rospy.Publisher = rospy.Publisher('vision/shape_detector/contours_image', Image, queue_size=10)

        self._lock.release()

    def start(self):
        rospy.spin()

    def _handle_img(self, msg):
        self._lock.acquire()

        img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        detections = FeatureDetections()
        detections.detector_tag = 'shape_detector'

        tf_cam_odom_H = None
        tf_cam_odom_quat = None
        world_frame = f'{self._tf_namespace}/odom'
        camera_frame = f'{self._tf_namespace}/{self._frame_id}'
        try:
            tf_cam_odom = self._tf_buffer.lookup_transform(
                world_frame,
                camera_frame,
                msg.header.stamp,
                rospy.Duration(0.1)
            )
            tf_cam_odom_H = tf2_transform_to_homogeneous(tf_cam_odom)
            tf_cam_odom_rotm = Rotation.from_quat(tf2_transform_to_quat(tf_cam_odom)).as_matrix()
        except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
            rospy.logwarn(f'Could not get transform from {world_frame} to {camera_frame}: {e}')
            self._lock.release()
            return

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = img_hsv.shape

        img_thresh = np.zeros((h, w), dtype=np.uint8)
        for hsv_range in self._hsv_ranges:
            low = np.array([hsv_range[0], hsv_range[2], hsv_range[4]])
            high = np.array([hsv_range[1], hsv_range[3], hsv_range[5]])
            mask = cv2.inRange(img_hsv, low, high)
            img_thresh = img_thresh | mask

        img_blur = cv2.GaussianBlur(img_thresh, (5, 5), 0)

        contours, _ = cv2.findContours(img_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours = [contour for contour in contours if cv2.contourArea(contour) >= self._contour_area_threshold]

        img_contours = img.copy()

        for contour in contours:
            ellipse = cv2.fitEllipse(contour)

            (e_y, e_x), (e_w, e_h), e_theta_deg = ellipse
            e_theta = -np.deg2rad(e_theta_deg)

            n_ellipse_contour_points = 32
            ellipse_contour = np.zeros((n_ellipse_contour_points, 2))

            alpha = np.linspace(0, 2 * pi, n_ellipse_contour_points)

            ellipse_contour[:, 0] = e_y + (e_h / 2) * np.cos(alpha) * sin(e_theta) + (e_w / 2) * np.sin(alpha) * cos(e_theta)
            ellipse_contour[:, 1] = e_x + (e_h / 2) * np.cos(alpha) * cos(e_theta) - (e_w / 2) * np.sin(alpha) * sin(e_theta)

            ellipse_defects = np.zeros((n_ellipse_contour_points))
            for i in range(n_ellipse_contour_points):
                ellipse_point = (ellipse_contour[i, 0], ellipse_contour[i, 1])
                ellipse_defects[i] = abs(cv2.pointPolygonTest(contour, ellipse_point, True))

            ellipse_defects_mean = np.mean(ellipse_defects) / min(e_w, e_h)

            cv2.ellipse(img_contours, ellipse, (255, 0, 0), 1)

            cv2.drawContours(img_contours, [ellipse_contour.astype(int)], -1, (255, 255, 0), 1)

            cv2.putText(img_contours, f'{ellipse_defects_mean:.4f}', (int(e_y), int(e_x)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            if ellipse_defects_mean > self._ellipse_defects_threshold:
                continue

            cv2.ellipse(img_contours, ellipse, (255, 0, 0), 1)

            flat_contour = contour.reshape((contour.shape[0], contour.shape[2]))

            max_idx = np.argmax(flat_contour, axis=0)
            min_idx = np.argmin(flat_contour, axis=0)

            center = np.array([ellipse[0][0], ellipse[0][1]])
            max_x, max_y = flat_contour[max_idx]
            min_x, min_y = flat_contour[min_idx]

            points_image = np.array([
                min_y,
                max_x,
                max_y,
                min_x,
                # center
            ]).astype(np.float64)

            c_w = self._circle_width
            points_world = np.array([
                [0, -c_w / 2, 0],
                [c_w / 2, 0, 0],
                [0, c_w / 2, 0],
                [-c_w / 2, 0, 0],
                # [0, 0, 0]
            ], dtype=np.float64)

            res, rvec, tvec = cv2.solvePnP(points_world, points_image, self._intrinsics, self._distortion, flags=cv2.SOLVEPNP_IPPE)

            axes_world = np.array([
                [0, 0, 0],
                [0.1, 0, 0],
                [0, 0.1, 0],
                [0, 0, 0.1],
            ], dtype="double")
            axes_image, _ = cv2.projectPoints(axes_world, rvec, tvec, self._intrinsics, self._distortion)
            # axes_image, _ = cv2.projectPoints(axes_world, rvec, tvec, self._intrinsics, self._distortion, cv2.SOLVEPNP_IPPE)
            axes_image = np.array(axes_image).astype(int)

            cv2.line(img_contours, tuple(axes_image[0,0,:]), tuple(axes_image[1,0,:]), color=(0, 0, 255), thickness=3)
            cv2.line(img_contours, tuple(axes_image[0,0,:]), tuple(axes_image[2,0,:]), color=(0, 255, 0), thickness=3)
            cv2.line(img_contours, tuple(axes_image[0,0,:]), tuple(axes_image[3,0,:]), color=(255, 0, 0), thickness=3)

            position_cam_h = np.array([tvec[0, 0], tvec[1, 0], tvec[2, 0], 1])
            position_odom_h = tf_cam_odom_H @ position_cam_h
            position_odom = position_odom_h[0:3] / position_odom_h[3]

            rotm_cam, _ = cv2.Rodrigues(rvec)

            yaw_cam = Rotation.from_matrix(rotm_cam).as_euler('ZYX')[1]

            yaw_cam = (yaw_cam + pi) % (2 * pi) - pi

            if yaw_cam < -0.4 or yaw_cam > 0.4:
                yaw_cam = 0
                # continue

            yaw_odom = yaw_cam + Rotation.from_matrix(tf_cam_odom_rotm).as_euler('ZYX')[0] - pi / 2

            detection = FeatureDetection()
            detection.position = Point(position_odom[0], position_odom[1], position_odom[2])
            detection.orientation = Point(0, 0, yaw_odom)
            detection.confidence = 1
            detection.SE2 = False
            detection.tag = 'circle'
            detections.detections.append(detection)

        contours_img_msg = self._cv_bridge.cv2_to_imgmsg(img_contours, encoding='bgr8')
        self._contours_img_pub.publish(contours_img_msg)

        self._detections_pub.publish(detections)

        self._lock.release()

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame_id = rospy.get_param('~frame_id')
        self._hsv_ranges = rospy.get_param('~hsv_ranges')
        self._contour_area_threshold = rospy.get_param('~contour_area_threshold')
        self._circle_width = rospy.get_param('~circle_width')
        self._ellipse_defects_threshold = rospy.get_param('~ellipse_defects_threshold')


def main():
    rospy.init_node('shape_detector')
    n = ShapeDetector()
    n.start()