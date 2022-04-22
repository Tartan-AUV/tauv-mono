import rospy
import numpy as np 
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from darknet_ros_msgs.msg import BoundingBoxes

# SHIT FOR TEST: WILL FIX LATER
class DepthEstimator: 
    def __init__(self):
        self.imageWidth = 1280
        self.imageHeight = 720
        self.numBits = 32
        self.maxVal = 2**self.numBits - 1
        
        # self.depth_image = np.zeros((self.imageHeight, self.imageWidth))
        self.depth_camera_info = CameraInfo()
        self.bounding_boxes = BoundingBoxes() 

        self.depth_image_streamer = rospy.Subscriber("/zedm_A/zed_node_A/depth/depth_registered", Image, self.depth_callback)
        self.depth_camera_info = rospy.Subscriber("/zedm_A/zed_node_A/left/camera_info", CameraInfo, self.camera_info_callback)
        self.bounding_boxes = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)

        self.cv_bridge = CvBridge()

        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)
        self.new_bbox = False
        self.new_image = False
        self.object_dict = dict()


    def camera_info_callback(self, msg):
      self.depth_camera_info = msg

    def depth_callback(self, msg):
      self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
      self.new_image = True

    def bbox_callback(self, msg):
      self.bounding_boxes = msg 
      self.new_bbox = True

    def spin(self, event):
      if(self.new_bbox and self.new_image):
        self.new_bbox, self.new_image = False, False

        fx = self.depth_camera_info.K[0]
        cx = self.depth_camera_info.K[2]
        fy = self.depth_camera_info.K[4]
        cy = self.depth_camera_info.K[5]
        bboxes = self.bounding_boxes.bounding_boxes
        
        for bbox in bboxes:
          center_x = (bbox.xmin + bbox.xmax) // 2
          center_y = (bbox.ymin + bbox.ymax) // 2
          cur_depth = self.estimate_depth(center_x, center_y, 5, bbox)

          if (cur_depth != np.nan):
            cur_x = ((center_x - cx) * cur_depth) / (fx)
            cur_y = ((center_y - cy) * cur_depth) / (fy)
            print(bbox.Class, cur_x, cur_y, cur_depth)
            self.object_dict[bbox.Class] = (cur_depth, cur_x, cur_y)
          else: 
            print(cur_depth)


    def estimate_depth(self, x, y, w, bbox):
      box = self.depth_image[max(bbox.ymin, y - w) : min(bbox.ymax, y + w+1), max(bbox.xmin, x - w) : min(bbox.xmax, x + w + 1)]
      return np.nanmean(box)
       
      


def main():
    rospy.init_node('depth_estimation', anonymous=True)
    my_depth_estimator = DepthEstimator()
    rospy.spin()