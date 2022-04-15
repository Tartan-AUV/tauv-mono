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
        

        # self.depth_image = np.zeros(self.imageWidth, self.imageHeight)
        self.depth_camera_info = CameraInfo()
        self.bounding_boxes = BoundingBoxes() 

        self.depth_image_streamer = rospy.Subscriber("/zedm/zed_node/depth/depth_registered", Image, self.depth_callback)
        self.depth_camera_info = rospy.Subscriber("/zedm/zed_node/depth/depth_registered", CameraInfo, self.camera_info_callback)
        self.bounding_boxes = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)

        self.cv_bridge = CvBridge()

        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)
        self.new_bbox = False


    def camera_info_callback(self, msg):
      self.depth_camera_info = msg

    def depth_callback(self, msg):
      self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")

    def bbox_callback(self, msg):
      self.bounding_boxes = msg 
      self.new_bbox = True

    def spin(self, event):
      if(self.new_bbox):
        self.new_bbox = False
        bboxes = self.bounding_boxes.bounding_boxes
        for box in bboxes:
          cur_depth = self.estimate_depth((box.xmin + box.xmax) // 2, (box.ymin + box.ymax) // 2)
          print(box.xmin, box.ymin, box.xmax, box.ymax, box.Class, cur_depth)
          


    def estimate_depth(self, x, y):
      return self.depth_image[y, x]
       
      


def main():
    rospy.init_node('depth_estimation', anonymous=True)
    my_depth_estimator = DepthEstimator()
    rospy.spin()