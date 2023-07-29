import rospy
import threading
from functools import partial
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
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from tauv_util.spatialmath import r3_to_ros_point
from tauv_util.cameras import CameraIntrinsics, CameraDistortion
import message_filters
from transform_client import TransformClient
from spatialmath import SE3
from .circle_detection import GetCirclePosesParams, get_circle_poses
from .path_marker_detection import GetPathMarkerPosesParams, get_path_marker_poses
from .adaptive_color_thresholding import GetAdaptiveColorThresholdingParams, get_adaptive_color_thresholding


class ShapeDetector:

    def __init__(self):
        self._lock: threading.Lock = threading.Lock()
        self._lock.acquire()

        self._load_config()

        self._tf_client: TransformClient = TransformClient()

        self._cv_bridge: CvBridge = CvBridge()

        self._camera_infos: {str: CameraInfo} = {}
        self._intrinsics: {str: CameraIntrinsics} = {}
        self._distortions: {str: CameraDistortion} = {}
        for frame_id in self._frame_ids:
            info = rospy.wait_for_message(f'vehicle/{frame_id}/color/camera_info', CameraInfo, 60)
            self._camera_infos[frame_id] = info
            self._intrinsics[frame_id] = CameraIntrinsics.from_matrix(np.array(info.K))
            self._distortions[frame_id] = CameraDistortion.from_matrix(np.array(info.D))

        self._synchronizers: {str: message_filters.ApproximateTimeSynchronizer} = {}
        self._circle_debug_img_pubs: {str: rospy.Publisher} = {}
        self._path_marker_debug_img_pubs: {str: rospy.Publisher} = {}
        for frame_id in self._frame_ids:
            color_sub = message_filters.Subscriber(f'vehicle/{frame_id}/color/image_raw', Image)
            depth_sub = message_filters.Subscriber(f'vehicle/{frame_id}/depth/image_raw', Image)

            synchronizer = message_filters.ApproximateTimeSynchronizer(
                [color_sub, depth_sub], queue_size=self._synchronizer_queue_size, slop=self._synchronizer_slop
            )
            synchronizer.registerCallback(partial(self._handle_imgs, frame_id=frame_id))
            self._synchronizers[frame_id] = synchronizer

            self._circle_debug_img_pubs[frame_id] =\
                rospy.Publisher(f'vision/shape_detector/{frame_id}/circle/debug_image', Image, queue_size=10)
            self._path_marker_debug_img_pubs[frame_id] = \
                rospy.Publisher(f'vision/shape_detector/{frame_id}/path_marker/debug_image', Image, queue_size=10)

        self._detections_pub: rospy.Publisher =\
            rospy.Publisher('global_map/feature_detections', FeatureDetections, queue_size=10)

        self._reload_config_server: rospy.Service = rospy.Service(f'vision/shape_detector/reload_config', Trigger, self._handle_reload_config)

        self._lock.release()

    def start(self):
        rospy.spin()

    def _handle_imgs(self, color_msg: Image, depth_msg: Image, frame_id: str):
        with self._lock:
            color = self._cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth = self._cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='mono16')
            depth = depth.astype(float) / 1000

            world_frame = f'{self._tf_namespace}/odom'
            camera_frame = f'{self._tf_namespace}/{frame_id}'

            world_t_cam = self._tf_client.get_a_to_b(world_frame, camera_frame, color_msg.header.stamp)

            detections = FeatureDetections()
            detections.header.stamp = color_msg.header.stamp
            detections.detector_tag = 'shape_detector'

            circle_detections, circle_debug_img = self._get_circle_detections(color, depth, world_t_cam,
                                                            self._intrinsics[frame_id])
            detections.detections = detections.detections + circle_detections
            circle_debug_msg = self._cv_bridge.cv2_to_imgmsg(circle_debug_img, encoding='bgr8')
            self._circle_debug_img_pubs[frame_id].publish(circle_debug_msg)

            path_marker_detections, path_marker_debug_img = self._get_path_marker_detections(color, depth, world_t_cam, self._intrinsics[frame_id])
            detections.detections = detections.detections + path_marker_detections
            path_marker_debug_msg = self._cv_bridge.cv2_to_imgmsg(path_marker_debug_img, encoding='bgr8')
            self._path_marker_debug_img_pubs[frame_id].publish(path_marker_debug_msg)

            self._detections_pub.publish(detections)

    def _handle_reload_config(self, req: TriggerRequest) -> TriggerResponse:
        res = TriggerResponse()

        with self._lock:
            self._load_config()

        res.success = True
        res.message = 'success'

        return res

    def _get_circle_detections(self, color: np.array, depth: np.array, world_t_cam: SE3,
                               intrinsics: CameraIntrinsics) -> ([FeatureDetection], np.array):
        mask = get_adaptive_color_thresholding(color, self._circle_threshold_params)

        debug_img = cv2.cvtColor(255 * mask, cv2.COLOR_GRAY2BGR)

        cam_t_circles = get_circle_poses(mask, depth, intrinsics, self._circle_pose_params, debug_img)

        world_t_circles = [world_t_cam * cam_t_circle for cam_t_circle in cam_t_circles]

        detections = []
        for world_t_circle in world_t_circles:
            detection = FeatureDetection()

            detection.count = 1
            detection.tag = 'circle'
            detection.position = r3_to_ros_point(world_t_circle.t)
            detection.orientation = r3_to_ros_point(world_t_circle.rpy())

            detections.append(detection)

        return detections, debug_img

    def _get_path_marker_detections(self, color: np.array, depth: np.array, world_t_cam: SE3,
                                    intrinsics: CameraIntrinsics) -> ([FeatureDetection], np.array):
        mask = get_adaptive_color_thresholding(color, self._path_marker_threshold_params)

        debug_img = cv2.cvtColor(255 * mask, cv2.COLOR_GRAY2BGR)

        cam_t_path_markers = get_path_marker_poses(mask, depth, intrinsics, self._path_marker_pose_params, debug_img)

        world_t_path_markers = [world_t_cam * cam_t_path_marker for cam_t_path_marker in cam_t_path_markers]

        detections = []
        for world_t_path_marker in world_t_path_markers:
            detection = FeatureDetection()

            detection.confidence = 1
            detection.tag = 'path_marker'
            detection.SE2 = False
            detection.position = r3_to_ros_point(world_t_path_marker.t)
            detection.orientation = r3_to_ros_point(world_t_path_marker.rpy())

            detections.append(detection)

        return detections, debug_img

    def _load_config(self):
        self._tf_namespace = rospy.get_param('tf_namespace')
        self._frame_ids = rospy.get_param('~frame_ids')
        self._synchronizer_queue_size = rospy.get_param('~synchronizer_queue_size')
        self._synchronizer_slop = rospy.get_param('~synchronizer_slop')

        self._circle_threshold_params: GetAdaptiveColorThresholdingParams = GetAdaptiveColorThresholdingParams(
            global_thresholds=np.array(rospy.get_param('~circle/threshold/global_thresholds')),
            local_thresholds=np.array(rospy.get_param('~circle/threshold/local_thresholds')),
            window_size=rospy.get_param('~circle/threshold/window_size'),
        )
        self._circle_pose_params: GetCirclePosesParams = GetCirclePosesParams(
            min_size=tuple(rospy.get_param('~circle/pose/min_size')),
            max_size=tuple(rospy.get_param('~circle/pose/max_size')),
            min_aspect_ratio=rospy.get_param('~circle/pose/min_aspect_ratio'),
            max_aspect_ratio=rospy.get_param('~circle/pose/max_aspect_ratio'),
            depth_mask_scale=rospy.get_param('~circle/pose/depth_mask_scale'),
        )

        self._path_marker_threshold_params: GetAdaptiveColorThresholdingParams = GetAdaptiveColorThresholdingParams(
            global_thresholds=np.array(rospy.get_param('~path_marker/threshold/global_thresholds')),
            local_thresholds=np.array(rospy.get_param('~path_marker/threshold/local_thresholds')),
            window_size=rospy.get_param('~path_marker/threshold/window_size'),
        )
        self._path_marker_pose_params: GetPathMarkerPosesParams = GetPathMarkerPosesParams(
            min_size=tuple(rospy.get_param('~path_marker/pose/min_size')),
            max_size=tuple(rospy.get_param('~path_marker/pose/max_size')),
            min_aspect_ratio=rospy.get_param('~path_marker/pose/min_aspect_ratio'),
            max_aspect_ratio=rospy.get_param('~path_marker/pose/max_aspect_ratio'),
        )


def main():
    rospy.init_node('shape_detector')
    n = ShapeDetector()
    n.start()