import rospy
import threading
import collections
import numpy as np
import tf2_ros as tf2
import cv_bridge
from sensor_msgs.msg import CameraInfo, Image
from tauv_msgs.msg import FeatureDetections, FeatureDetection
from tauv_util.transforms import tf2_transform_to_homogeneous, tf2_transform_to_quat, multiply_quat, quat_to_rpy
from geometry_msgs.msg import Point, Quaternion
from chevron_detector_processing import *
from scipy.spatial.transform import Rotation


class ChevronDetector:

    def __init__(self):
        self._lock: threading.Lock = threading.Lock()
        self._lock.acquire()

        self._load_config()

        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer)

        self._cv_bridge: cv_bridge.CvBridge = cv_bridge.CvBridge()

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

        detections = FeatureDetections()
        detections.detector_tag = 'chevron_detector'

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

        img_threshold = threshold(img, self._hsv_ranges)
        img_clean = clean(img_threshold)

        suction_mask = get_border_mask(img_clean, 0, self._suction_mask_width, 0, 0)
        img_clean = img_clean & suction_mask

        labels, n_labels = get_components(img_clean, self._area_threshold)

        valid_mask = get_border_mask(
            img_clean,
            self._border_mask_width,
            self._suction_mask_width + self._border_mask_width,
            self._border_mask_width,
            self._border_mask_width
        )

        for i in range(n_labels):
            component = np.where(labels == i, 1, 0)

            contour = get_contour(component, self._contour_approximation_factor)
            angles = get_angles(contour)
            classifications = classify_angles(contour, angles, self._angle_threshold, valid_mask)

            classifications_counter = collections.Counter(classifications)

            use_pnp = classifications_counter[AngleClassification.FRONT.value] == 1 \
                and classifications_counter[AngleClassification.BACK.value] == 1 \
                and classifications_counter[AngleClassification.SIDE.value] == 2 \

            use_fallback = classifications_counter[AngleClassification.FRONT.value] == 1

            if use_pnp:
                front_point = contour[classifications == AngleClassification.FRONT.value, :]
                back_point = contour[classifications == AngleClassification.BACK.value, :]
                side_points = contour[classifications == AngleClassification.SIDE.value, :]
                side_point_0 = side_points[0, :]
                side_point_1 = side_points[0, :]

                img_points = np.array([
                    front_point,
                    side_point_0,
                    side_point_1,
                    back_point,
                ]).astype(np.float64)

                res, rvec, tvec = cv2.solvePnP(self._points, img_points, self._intrinsics, self._distortion, flags=0)

                position_cam_h = np.array([tvec[0, 0], tvec[1, 0], tvec[2, 0], 1])
                position_odom_h = tf_cam_odom_H @ position_cam_h
                position_odom = position_odom_h[0:3] / position_odom_h[3]

                detection = FeatureDetection()
                detection.position = Point(position_odom[0], position_odom[1], position_odom[2])
                detection.orientation = Point(0, 0, 0)
                detection.count = 1
                detection.tag = 'chevron'
                detections.detections.append(detection)
            elif use_fallback:
                pass

        self._detections_pub.publish(detections)

        self._lock.release()

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame_id = rospy.get_param('~frame_id')
        self._hsv_ranges = rospy.get_param('~hsv_ranges')
        self._area_threshold = rospy.get_param('~area_threshold')
        self._suction_mask_width = rospy.get_param('~suction_mask_width')
        self._border_mask_width = rospy.get_param('~border_mask_width')
        self._contour_approximation_factor = rospy.get_param('~contour_approximation_factor')
        self._angle_threshold = rospy.get_param('~angle_threshold')
        self._points = rospy.get_param('~points')
        self._fallback_depth = rospy.get_param('~fallback_depth')


def main():
    rospy.init_node('chevron_detector')
    n = ChevronDetector()
    n.start()