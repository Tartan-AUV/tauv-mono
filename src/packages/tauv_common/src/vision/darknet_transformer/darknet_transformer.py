#!/usr/bin/env python3

import rospy
import threading
from typing import Dict
from tauv_msgs.msg import FeatureDetection, FeatureDetections
from tauv_msgs.srv import GetCameraInfo
import numpy as np
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image, CameraInfo
from vision.depth_estimation.depth_estimation import DepthEstimator
from tauv_util.transforms import tf2_transform_to_homogeneous
from geometry_msgs.msg import Vector3
import tf2_ros as tf2
from collections import deque


class DarknetTransformer():

    def __init__(self):
        self._load_config()

        self._depth_lock: threading.Lock = threading.Lock()
        self._bboxes_lock: threading.Lock = threading.Lock()
        self._depth_lock.acquire()
        self._bboxes_lock.acquire()

        self._tf_buffer: tf2.Buffer = tf2.Buffer()
        self._tf_listener: tf2.TransformListener = tf2.TransformListener(self._tf_buffer)

        self._cv_bridge: CvBridge = CvBridge()

        self._camera_infos: Dict[str, CameraInfo] = {}
        self._depth_subs: Dict[str, rospy.Subscriber] = {}

        self._bboxes_queues: Dict[str, deque[BoundingBoxes]] = {}
        self._depth_queues: Dict[str, deque[Image]] = {}

        self._bboxes_sub: rospy.Subscriber = rospy.Subscriber('darknet_ros/bounding_boxes', BoundingBoxes, self._handle_bboxes)
        self._detections_pub: rospy.Publisher = rospy.Publisher('global_map/feature_detections', FeatureDetections, queue_size=10)

        for frame_id in self._frame_ids:
            camera_info_topic = f'vehicle/{frame_id}/camera_info'
            rospy.wait_for_service(camera_info_topic, 10)
            camera_info_srv = rospy.ServiceProxy(camera_info_topic, GetCameraInfo)

            resp = camera_info_srv.call(frame_id)
            self._camera_infos[frame_id] = resp.camera_info

            self._depth_subs[frame_id] = rospy.Subscriber(f'vehicle/{frame_id}/depth', Image, self._handle_depth, callback_args=frame_id)

            self._bboxes_queues[frame_id] = deque(maxlen=self._bboxes_queue_size)
            self._depth_queues[frame_id] = deque(maxlen=self._depth_queue_size)

        print(self._bboxes_queues, self._depth_queues)
        self._depth_lock.release()
        self._bboxes_lock.release()


    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _handle_depth(self, msg, frame_id):
        self._depth_lock.acquire()
        self._depth_queues[frame_id].append(msg)
        self._depth_lock.release()

    def _handle_bboxes(self, msg):
        self._bboxes_lock.acquire()
        frame_id = msg.image_header.frame_id.split('/')[1]
        self._bboxes_queues[frame_id].append(msg)
        self._bboxes_lock.release()

    def _update(self, timer_event):
        self._depth_lock.acquire()
        self._bboxes_lock.acquire()

        # TODO: needs something to remove bboxes after processing

        for frame_id in self._frame_ids:
            bboxes_queue = self._bboxes_queues[frame_id]
            depth_queue = self._depth_queues[frame_id]

            for (bboxes_i, bboxes) in enumerate(bboxes_queue):
                deltas =\
                    [(depth_i, (msg.header.stamp - bboxes.image_header.stamp).to_sec())for (depth_i, msg) in enumerate(depth_queue)]
                print(deltas)
                deltas = list(filter(lambda t : abs(t[1]) < self._max_sync_delta, deltas))

                if len(deltas) == 0:
                    continue

                matched_depth_i, matched_depth_delta = min(deltas, key=lambda t : t[1])
                matched_depth = depth_queue[matched_depth_i]

                self._transform(frame_id, bboxes, matched_depth)

        self._depth_lock.release()
        self._bboxes_lock.release()

    def _transform(self, frame_id, bboxes, depth):
        print('transform')
        detections = FeatureDetections()
        detections.header.frame_id = frame_id
        detections.header.stamp = depth.header.stamp
        detections.header.seq = depth.header.seq

        depth_img = self._cv_bridge.imgmsg_to_cv2(depth, desired_encoding='mono16')

        for bbox in bboxes.bounding_boxes:
            bbox_depth = DepthEstimator.estimate_absolute_depth(depth_img, bbox, self._camera_infos[frame_id])

            if bbox_depth == np.nan:
                continue

            # TODO: Change this
            world_frame = f'{self._tf_namespace}/odom'
            camera_frame = f'{self._tf_namespace}/{frame_id}'

            try:
                transform = self._tf_buffer.lookup_transform(
                    world_frame,
                    camera_frame,
                    depth.header.stamp,
                )

                H = tf2_transform_to_homogeneous(transform)

                world_point_h = H @ np.array([bbox_depth[1], bbox_depth[2], bbox_depth[0], 1])
                world_point = world_point_h[0:3] / world_point_h[3]

                detection = FeatureDetection()
                detection.tag = bbox.Class
                detection.position = Vector3(world_point[0], world_point[1], world_point[2])
                detections.detections.append(detection)
            except (tf2.LookupException, tf2.ConnectivityException, tf2.ExtrapolationException) as e:
                rospy.logerr(f'Could not get transform from {world_frame} to {camera_frame}: {e}')

        self._detections_pub.publish(detections)

    def _load_config(self):
        self._frame_ids: [str] = rospy.get_param('~frame_ids')
        self._tf_namespace: str = rospy.get_param('tf_namespace')
        self._bboxes_queue_size = 10
        self._depth_queue_size = 10
        self._dt = 1 / 10
        self._max_sync_delta = 10.0
        self._max_tf_delta = 10.0

def main():
    rospy.init_node('darknet_transformer')
    n = DarknetTransformer()
    n.start()