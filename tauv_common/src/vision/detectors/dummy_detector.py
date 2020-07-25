# dummy_detector
#
# This node is for testing the vision bucket registration services
#
#
# Author: Advaith Sethuraman 2020

#!/usr/bin/env python
import rospy
import tf
import tf_conversions
import numpy as np
import itertools
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Imu, Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import *
from jsk_recognition_msgs.msg import BoundingBox
from nav_msgs.msg import Odometry
from tf.transformations import *
from std_msgs.msg import *
from geometry_msgs.msg import Quaternion
from tauv_msgs.msg import BucketDetection, BucketList
from tauv_common.srv import RegisterObjectDetection
from scipy.spatial.transform import Rotation as R


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

class Dummy_Detector():
    def __init__(self):
        rospy.wait_for_service("detector_bucket/register_object_detection")
        self.registration_service = rospy.ServiceProxy("detector_bucket/register_object_detection", RegisterObjectDetection)
        self.left_stream = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_image", Image, self.left_callback)
        self.disparity_stream = rospy.Subscriber("/vision/front/disparity", DisparityImage, self.disparity_callback)
        self.left_camera_info = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_info", CameraInfo, self.camera_info_callback)
        self.left_camera_detections = rospy.Publisher("front_detections", Image, queue_size=10)

        self.weights = "/home/advaiths/foreign_disk/catkin_robosub/src/TAUV-ROS-Packages/tauv_common/src/vision/detectors/yolov3.weights"
        self.config = "/home/advaiths/foreign_disk/catkin_robosub/src/TAUV-ROS-Packages/tauv_common/src/vision/detectors/yolov3.cfg"
        self.classes_list = "/home/advaiths/foreign_disk/catkin_robosub/src/TAUV-ROS-Packages/tauv_common/src/vision/detectors/yolov3.txt"
        self.arrow_list = []
        self.classes = self.prepare_classes()
        self.stereo_proc = cv2.StereoBM_create(numDisparities=16, blockSize=33)
        self.net = self.load_model()
        self.tf = tf.TransformListener()
        self.cv_bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.orb = cv2.ORB_create(300)
        self.focal_length = 0.0
        self.baseline = 0.0
        self.stereo_left = Image()
        self.stereo_right = []
        self.disparity = DisparityImage()
        self.baseline = 0
        self.left_camera_info = CameraInfo()
        self.left_img_flag = False
        self.disp_img_flag = False
        self.rate = rospy.Rate(1)
        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)
        self.marker_id = 0

    # function to get the output layer names
    # in the architecture
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = [0, 0, 255]
        cv2.rectangle(img, (int(x), int(y)), (int(x_plus_w), int(y_plus_h)), color, 2)
        cv2.putText(img, label, (int(x-10), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def load_model(self):
        return cv2.dnn.readNet(self.weights, self.config)

    def prepare_classes(self):
        classes = None
        with open(self.classes_list, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def classify(self, image):
        now = rospy.Time(0)
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        final_detections = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            final_detections.append([class_ids[i], (x, y, w, h)])
            self.draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        self.left_camera_detections.publish(self.cv_bridge.cv2_to_imgmsg(image))
        return final_detections, now

    def camera_info_callback(self, msg):
        self.left_camera_info = msg

    def disparity_callback(self, msg):
        self.disp_img_flag = True
        self.disparity = msg
        self.baseline = msg.T

    def left_callback(self, msg):
        self.stereo_left = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.left_img_flag = True

    def query_depth_map(self, map, pixel):
        return map[pixel[:,1], pixel[:,0]]

    def query_depth_map_rectangle(self, map, bbox_detection):
        x, y, w, h = bbox_detection[1]
        return map[y:y+h, x:x+w].reshape(-1, 1)

    def depth_from_disparity(self, msg):
        self.focal_length = msg.f
        self.baseline = msg.T
        disparity_image = self.cv_bridge.imgmsg_to_cv2(msg.image, "passthrough")
        return self.focal_length * self.baseline / disparity_image

    def vector_to_detection_centroid(self, bbox_detection):
        x, y, w, h = bbox_detection[1]
        disp_map = self.cv_bridge.imgmsg_to_cv2(self.disparity.image, "passthrough")
        x_cnt = x+w/2
        y_cnt = y+h/2
        d = disp_map[y_cnt, x_cnt]
        d = np.mean(self.query_depth_map_rectangle(disp_map, bbox_detection))

        centroid_2d = np.asmatrix([x_cnt, y_cnt, d, 1]).T

        K = np.asarray(self.left_camera_info.K).reshape((3, 3))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        Q = np.asmatrix([[1, 0, 0, -cx],
                         [0, 1, 0, -cy],
                         [0, 0, 0, fx],
                         [0, 0, -1.0/self.baseline, 0]])
        centroid_3d = Q * centroid_2d
        centroid_3d /= centroid_3d[3]
        return centroid_3d[0:3]

    def prepare_detection_registration(self, centroid, det, now):
        obj_det = BucketDetection()
        obj_det.image = self.cv_bridge.cv2_to_imgmsg(self.stereo_left, "bgr8")
        obj_det.tag = str("vision/object_tags/" + self.classes[det[0]])
        bbox_3d = BoundingBox()
        bbox_3d.dimensions = Vector3(.25, .25, 1.0)
        bbox_pose = Pose()
        #print(feature_centroid.shape)
        x, y, z = list((np.squeeze(centroid)).T)
        #print(x, y, z)
        obj_det.position = Point(x, y, z)
        bbox_pose.position = Point(x, y, z)
        bbox_3d.pose = bbox_pose
        bbox_header = Header()
        bbox_header.frame_id = "duo3d_optical_link_front"
        bbox_header.stamp = now
        bbox_3d.header = bbox_header
        obj_det.bbox_3d = bbox_3d
        obj_det.header = Header()
        obj_det.header.frame_id = bbox_header.frame_id
        obj_det.header.stamp = now
        return obj_det

    def spin(self, event):
        if(self.left_img_flag and self.disp_img_flag):
            self.left_img_flag = False
            self.disp_img_flag = False
            detections, now = self.classify(self.stereo_left)
            for det in detections:
                feature_centroid = self.vector_to_detection_centroid(det)
                obj_det = self.prepare_detection_registration(feature_centroid, det, now)
                success = self.registration_service(obj_det)


        # if(len(keypoints) > 100):
        #     feature_centroid = self.get_feature_centroid(self.depth_from_disparity(self.disparity), keypoints)
        #     cv2.imwrite("/home/advaith/Desktop/object_detection_test.png", self.stereo_left)
        #     cv2.imshow("Features", cv2.drawKeypoints(self.stereo_left, keypoints, None))
        #     cv2.waitKey(1)
        #     self.left_img_flag = False
        #     obj_det = BucketDetection()
        #     obj_det.image = self.cv_bridge.cv2_to_imgmsg(self.stereo_left, "bgr8")
        #     obj_det.tag = "Testing"
        #     bbox_3d = BoundingBox()
        #     bbox_3d.dimensions = Vector3(1, 1, 1)
        #     now = rospy.Time(0)
        #     self.tf.waitForTransform("/odom", "/duo3d_left_link_front", now, rospy.Duration(4.0))
        #     (trans, rot) = self.tf.lookupTransform("/odom", "/duo3d_left_link_front", now)
        #     bbox_pose = Pose()
        #     x, y, z = feature_centroid + np.asarray(trans)
        #     obj_det.position = Point(x, y, z)
        #     bbox_pose.position = Vector3(x, y, z)
        #     bbox_3d.pose = bbox_pose
        #     bbox_header = Header()
        #     bbox_header.frame_id = "odom"
        #     bbox_3d.header = bbox_header
        #     obj_det.bbox_3d = bbox_3d
        #     success = self.registration_service(obj_det)
        #     print("Detection transmitted: " + str(success))

def main():
    rospy.init_node("dummy_detector")
    dummy_detector = Dummy_Detector()
    rospy.spin()