import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

import numpy as np
import torch
from spatialmath import SE3, SO3
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from functools import partial
from typing import Dict
import pathlib
from math import pi
import torchvision.transforms as T
import cv2
from threading import Lock
import platform

#from transform_client import TransformClient

from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from threading import Event

from tauv_util.cameras import CameraIntrinsics
from tauv_util.spatialmath import ros_transform_to_se3, r3_to_ros_vector3, r3_to_ros_point
from tauv_msgs.msg import FeatureDetection, FeatureDetections
from centernet.model.backbones.centerpoint_dla import CenterpointDLA34
from centernet.model.config import ObjectConfig, ObjectConfigSet, AngleConfig, ModelConfig, TrainConfig
from centernet.model.decode import decode_keypoints

from centernet.configs.samples_torpedo_bin_buoy import model_config, train_config, object_config


# Need to add frames for the banner
object_t_detections: Dict[str, SE3] = {
    # "torpedo_24": SE3(SO3.TwoVectors(x="-z", y="x")),
    # "torpedo_24_octagon": SE3(SO3.TwoVectors(x="-z", y="x")),
}

# weights_path = pathlib.Path("/shared/weights/dauntless-disco-272-latest.pt").expanduser()


# Get new weights
weights_name = 'breezy-yoghurt-1521-latest.pt'

if platform.machine() == 'aarch64':
    weights_path = pathlib.Path(f'/shared/weights/{weights_name}')
else:
    weights_path = pathlib.Path(f"~/catkin_ws/weights/{weights_name}").expanduser()

class CenternetNode(Node):
    def __init__(self):
        super().__init__("centernet_node")
        self._load_config()

        #self._tf_client: TransformClient = TransformClient()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._cv_bridge: CvBridge = CvBridge()

        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            self.get_logger().info("No Cuda Available")

        self._centernet = CenterpointDLA34(object_config).to(self._device)
        self._centernet.load_state_dict(torch.load(weights_path, map_location=self._device))
        self._centernet.eval()

        self._centernet.forward(torch.rand(1, 3, model_config.in_h, model_config.in_w, device=self._device))

        self._camera_infos: Dict[str, CameraInfo] = {}
        self._intrinsics: Dict[str, CameraIntrinsics] = {}

        self.qos = QoSProfile(depth = 10)
        self._color_subs: Dict[str, rclpy.subscription.Subscription] = {}
        #self._depth_subs: Dict[str, rospy.Subscriber] = {}
        
        self._debug_pubs: Dict[str, rclpy.publisher.Publisher] = {}

        for frame_id in self._frame_ids:
            self._camera_infos[frame_id] = wait_for_message(self, f"vehicle/{frame_id}/depth/camera_info", CameraInfo, timeout=60)
            self._intrinsics[frame_id] = CameraIntrinsics.from_matrix(np.array(self._camera_infos[frame_id].K))

            self._color_subs[frame_id] = self.create_subscription(
                Image,
                f"vehicle/{frame_id}/color/image_raw",
                partial(self._handle_img, frame_id=frame_id),
                qos_profile= self.qos
)

            #self._debug_pubs[frame_id] = rospy.Publisher(f"centernet/{frame_id}/debug", Image, queue_size=10)

        # self._detections_pub: rospy.Publisher = rospy.Publisher("global_map/feature_detections", FeatureDetections, queue_size=10)

        self._detections_pub = self.create_publisher(msg_type = FeatureDetections, topic = 'global_map/feature_detections', qos_profile = self.qos)
    
    def start(self):
        rclpy.spin(self)


    def _handle_img(self, color_msg: Image, frame_id: str):
        color_np = self._cv_bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")

        img_height, img_width, _ = color_np.shape

        img_raw = T.ToTensor()(color_np)
        img = T.Resize((model_config.in_h, model_config.in_w))(img_raw.unsqueeze(0))
        img = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img).to(self._device)

        prediction = self._centernet.forward(img)

        # Assume given when using sim
        intrinsics = self._intrinsics[frame_id]

        M_projection = np.array([
            [intrinsics.f_x, 0, intrinsics.c_x],
            [0, intrinsics.f_y, intrinsics.c_y],
            [0, 0, 0],
        ])

        detections = decode_keypoints(
            prediction,
            model_config,
            object_config,
            M_projection,
            n_detections=10,
            keypoint_n_detections=50,
            score_threshold=0.6,
            keypoint_score_threshold=0.3,
            keypoint_angle_threshold=0.3,
        )[0]

        world_frame = f"{self._tf_namespace}/odom"
        camera_frame = f"{self._tf_namespace}/{frame_id}"

        world_t_cam = None
        # while world_t_cam is None:
        try:
            world_t_cam = self.tf_buffer.lookup_transform(world_frame, camera_frame, Time.from_msg(color_msg.header.stamp))
        except Exception as e:
            self.get_logger().warn(e)
            self.get_logger().warn("Failed to get transform")
            return

        self.get_logger().info("Got transforms")

        detection_debug_np = color_np.copy()

        detection_array_msg = FeatureDetections()
        detection_array_msg.detector_tag = "centernet"

        for detection_i, detection in enumerate(detections):
            cv2.circle(detection_debug_np, (int(detection.x * img_width), int(detection.y * img_height)), 3, (255, 0, 0), -1)

            e_x = detection.x * img_width
            e_y = detection.y * img_height
            w = detection.w * img_width
            h = detection.h * img_height

            # No depth, can't calculate
            # cv2.rectangle(
            #     depth_mask,
            #     (int(e_x - 0.4 * w), int(e_y - 0.4 * h)),
            #     (int(e_x + 0.4 * w), int(e_y + 0.4 * h)),
            #     255,
            #     -1
            # )

            cv2.rectangle(
                detection_debug_np,
                (int(e_x - 0.4 * w), int(e_y - 0.4 * h)),
                (int(e_x + 0.4 * w), int(e_y + 0.4 * h)),
                (0, 0, 255),
                1
            )

            cv2.putText(detection_debug_np, f"{detection.score:02f}", (int(e_x - 0.4 * w), int(e_y - 0.5 * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)


            # z = np.mean(depth[(depth_mask > 0) & (depth > 0)])

            # print(f"Mean z for detection: {z : .4f} m")

            # if z < 1:
            #     continue

            # print(f"f_x: {M_projection[0, 0]},  f_y: {M_projection[1, 1]},  c_x: {M_projection[0, 2]},  c_y: {M_projection[1, 2]}")

            # x = (e_x - M_projection[0, 2]) * (z / M_projection[0, 0])
            # y = (e_y - M_projection[1, 2]) * (z / M_projection[1, 1])

            # cam_t_detection = SE3.Rt(SO3.TwoVectors(x="z", y="x"), np.array([x, y, z]))

            detection_id = object_config.configs[detection.label].id
            # cam_t_detection = cam_t_object * object_t_detections[detection_id]

            # world_t_detection = world_t_cam * cam_t_detection
            # cam_t_object = detection.cam_t_object
            # cam_t_object = SE3.Rt(SO3.TwoVectors(x="z", y="x"), cam_t_object.t)

            # world_t_object = world_t_cam * cam_t_object
            # world_t_detection = world_t_object * object_t_detections[detection_id]
            # world_t_detection = world_t_object

            # self._tf_client.set_a_to_b('kf/odom', 'raw_buoy', world_t_object)
            # self._tf_client.set_a_to_b('kf/odom', 'adjusted_buoy', world_t_detection)

            # detection_msg = FeatureDetection()
            # detection_msg.confidence = 1
            # detection_msg.tag = detection_id
            # detection_msg.SE2 = False
            # detection_msg.position = r3_to_ros_point(world_t_detection.t)
            # rpy = world_t_detection.rpy()
            # detection_msg.orientation = r3_to_ros_point(rpy)

            # detection_array_msg.detections.append(detection_msg)

        self._detections_pub.publish(detection_array_msg)

        detection_debug_msg = self._cv_bridge.cv2_to_imgmsg(np.flip(detection_debug_np, axis=-1), encoding="bgr8")
        self._debug_pubs[frame_id].publish(detection_debug_msg)

    def _load_config(self):
        # self._frame_ids: [str] = rospy.get_param("~frame_ids")
        # self._tf_namespace: str = rospy.get_param("tf_namespace")
        self.declare_parameter("frame_ids", [])
        self.declare_parameter("tf_namespace", "") 
        self._frame_ids: list[str] = self.get_parameter("frame_ids").get_parameter_value().string_array_value
        self._tf_namespace: str = self.get_parameter("tf_namespace").get_parameter_value().string_value



def wait_for_message(node: Node, topic: str, msg_type, timeout: float = None):
    msg_container = {"msg": None}
    received_event = Event()

    def callback(msg):
        msg_container["msg"] = msg
        received_event.set()

    qos = QoSProfile(depth=1)
    sub = node.create_subscription(msg_type, topic, callback, qos)

    if not received_event.wait(timeout):
        node.destroy_subscription(sub)
        raise TimeoutError(f"No message received on {topic} after {timeout}s")

    node.destroy_subscription(sub)
    return msg_container["msg"]


def main():
    rclpy.init()
    node = CenternetNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()