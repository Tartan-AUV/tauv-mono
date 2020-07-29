#!/usr/bin/env python
import cv2
import numpy as np 
import sys
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


class gateDetector: 
    def __init__(self):
        self.numBits = 8
        self.imageWidth = 640
        self.imageHeight = 480
        self.maxVal = 2**self.numBits - 1
        self.gate_dimensions = np.array(rospy.get_param("object_tags/gate/dimensions")).astype(float)
        self.gate_width = self.gate_dimensions[0]
        self.gate_height = self.gate_dimensions[2]

        self.left_img_flag = False
        self.stereo_left = Image()
        self.left_camera_info = CameraInfo()
        self.left_stream = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_image", Image, self.left_callback)
        self.left_camera_info = rospy.Subscriber("/albatross/stereo_camera_left_front/camera_info", CameraInfo, self.camera_info_callback)

        self.cv_bridge = CvBridge()
        self.gate_detection_pub = rospy.Publisher("gate_detections", Image, queue_size=10)
        self.spin_callback = rospy.Timer(rospy.Duration(.010), self.spin)
        rospy.wait_for_service("detector_bucket/register_object_detection")
        self.registration_service = rospy.ServiceProxy("detector_bucket/register_object_detection", RegisterObjectDetection)
        self.prev = [None, None]

    def openImage (self, path):
        img = cv2.imread(path)
        self.imageWidth, self.imageHeight, _ = img.shape
        return img
    # Gate in competition is usually orange in a blue water environment 
    # Converting to YUV (YCbCr) allows us to enhance the warm V (red/orange)
    # channel of the image and tone down the cool U (blue) channel
    def enhanceRedChroma(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        Y, U, V = cv2.split(image)  
        scaleU = 1
        scaleV = 3
        midPoint = (2**(self.numBits-1)) - 1
        meanU = U.mean()
        offsetU = midPoint-scaleU*meanU 
        newU = cv2.convertScaleAbs(U, alpha=scaleU, beta=offsetU)
        meanV = V.mean()
        offsetV = midPoint-scaleV*meanV  

        newV = cv2.convertScaleAbs(V, alpha=scaleV, beta=offsetV)
        new_image = cv2.merge([Y, newU, newV])
        new_image = cv2.cvtColor(new_image,cv2.COLOR_YUV2BGR)
        return new_image
    
    def overlayGateDetection(self, img1, leftBar, rightBar):
        width, height, _ = img1.shape
        leftLineTop = leftBar, 0
        leftLineBottom = leftBar, height
        rightLineTop = rightBar, 0
        rightLineBottom = rightBar, height
        centerLineX = (leftBar + rightBar) // 2
        centerLineTop = centerLineX, 0
        centerLineBottom = centerLineX, height
        cv2.line(img1, leftLineTop, leftLineBottom, (0, 255, 0), thickness=2)
        cv2.line(img1, rightLineTop, rightLineBottom, (0, 255, 0), thickness=2)
        cv2.line(img1, centerLineTop, centerLineBottom, (255, 0, 0), thickness=2)
        pixel_height = np.ceil((rightBar - leftBar)/self.gate_width*self.gate_height)
        end_coord = (int(rightLineTop[0]), int(rightLineTop[1] + pixel_height))
        cv2.rectangle(img1, leftLineTop, end_coord, (0, 255, 0), thickness=2)
        return img1
    # Amplify all of the channels in the image allowing for greater visibility 
    # in marine environments 
    def increaseContrast(self,image):
        scale = 2
        midPoint = (2**(self.numBits-1)) - 1
        B, G, R = cv2.split(image)

        meanB = B.mean()
        offsetB = midPoint-scale*meanB 
        newB = cv2.convertScaleAbs(B, alpha=scale, beta=offsetB)

        meanG = G.mean()
        offsetG = midPoint-scale*meanG 
        newG = cv2.convertScaleAbs(G, alpha=scale, beta=offsetG)

        meanR = R.mean()
        offsetR = midPoint-scale*meanR  
        newR = cv2.convertScaleAbs(R, alpha=scale, beta=offsetR)

        new_image = cv2.merge([newB, newG, newR])
        return new_image
    # Before thresholding, apply a series of blurs to simplify and generalize 
    # the image. Afterwards, apply a series of dilations/erosions making the 
    # binarization more homogenous.  
    def getBinary(self, img):
        blurDim = self.imageHeight//8
        if blurDim % 2 == 0: 
            blurDim = blurDim + 1
        blurImg = cv2.medianBlur(img, blurDim)

        colorBinaryImg = self.maxVal-cv2.absdiff(img, blurImg)
        gaussDim = blurDim//2
        if gaussDim % 2 == 0: 
            gaussDim = gaussDim + 1       
        gaussBlurImg = cv2.GaussianBlur(colorBinaryImg, (gaussDim, gaussDim), 0)
        grayImg = cv2.cvtColor(gaussBlurImg, cv2.COLOR_BGR2GRAY)
        
        threshImg = cv2.adaptiveThreshold(grayImg, self.maxVal,
             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, gaussDim, 2)
        binImg = cv2.bitwise_not(threshImg)
        binImg = cv2.dilate(binImg,np.ones((5,1)),iterations = 2)
        binImg = cv2.erode(binImg,np.ones((5,1)),iterations = 2)
        binImg = cv2.dilate(binImg,np.ones((1,3)),iterations = 2)
        binImg = cv2.erode(binImg,np.ones((1,3)),iterations = 2)
        return binImg
    #Uses the binarized image and finds the column sum index with the minimum sum
    # indicating where the posts will be. 
    def getBars(self, img):
        barWidth = self.imageWidth//60
        columnSum = np.sum(img, axis = 0)
        firstBar = -1 
        firstMinVal = float('inf')
        for i, elem in enumerate(columnSum):
            if elem < firstMinVal: 
                firstBar = i 
                firstMinVal = elem 
        numWhite = firstMinVal/self.maxVal 
        fillFactor = numWhite/self.imageHeight 
        if fillFactor > 0.80:
            return (None, None)
        secondMinVal = float('inf')
        for i, elem in enumerate(columnSum):
            if elem < secondMinVal and abs(i - firstBar) > barWidth: 
                secondBar = i 
                secondMinVal = elem 
        if firstBar > secondBar: 
            firstBar, secondBar = secondBar, firstBar
        if abs(secondBar - firstBar) < self.imageWidth // 4:
            return (None, None)
        return (firstBar, secondBar)

    def findPost(self, img):
        now = rospy.Time(0)
        (height, width, channels) = img.shape
        self.imageHeight = height 
        self.imageWidth = width
        yuvImg = self.enhanceRedChroma(img)
        contrastImg = self.increaseContrast(yuvImg)
        binaryImg = self.getBinary(contrastImg)
        (leftBar, rightBar) = self.getBars(binaryImg)
        return (leftBar, rightBar, now)

    def camera_info_callback(self, msg):
        self.left_camera_info = msg

    def left_callback(self, msg):
        self.stereo_left = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        self.left_img_flag = True

    def prepareDetectionRegistration(self, centroid, now):
        obj_det = BucketDetection()
        obj_det.image = self.cv_bridge.cv2_to_imgmsg(self.stereo_left, "bgr8")
        obj_det.tag = "object_tags/gate"
        bbox_3d = BoundingBox()
        bbox_3d.dimensions = Vector3(self.gate_dimensions[0], self.gate_dimensions[1], self.gate_dimensions[2])
        bbox_pose = Pose()
        x, y, z = list((np.squeeze(centroid)).T)
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

    def vector_to_detection_centroid(self, leftBar, rightBar):
        centerX = (leftBar+rightBar)//2
        K = np.asarray(self.left_camera_info.K).reshape((3, 3))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        z = float(self.gate_width)*fx/float(abs(rightBar - leftBar))
        x = float(centerX - cx)/float(abs(rightBar - leftBar))*float(self.gate_width)
        centroid_3d = np.asmatrix([x, -cy, z, 1]).T
        # centroid_3d /= centroid_3d[3]
        return centroid_3d[0:3]

    def spin(self, event):
        if(self.left_img_flag):
            self.left_img_flag = False
            leftBar, rightBar, now = self.findPost(self.stereo_left)
            overlayedImage = self.stereo_left
            if leftBar != None and rightBar != None:
                if self.prev[0] == None or self.prev[1] == None:
                    self.prev = (leftBar, rightBar)
                else:
                    leftBar, rightBar = (int((self.prev[0] + leftBar)//2), int((self.prev[1] + rightBar)//2))
                    self.prev = (leftBar, rightBar)
                overlayedImage = self.overlayGateDetection(self.stereo_left, leftBar, rightBar)
                centroid = self.vector_to_detection_centroid(leftBar, rightBar)
                obj_det = self.prepareDetectionRegistration(centroid, now)
                success = self.registration_service(obj_det)
            self.gate_detection_pub.publish(self.cv_bridge.cv2_to_imgmsg(overlayedImage))

def main():
    rospy.init_node('gate_detector', anonymous=True)
    myGateDetector = gateDetector()
    rospy.spin()

if __name__ == '__main__':
    main()