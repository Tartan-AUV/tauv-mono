import rospy

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
#from tauv_msgs.srv import HSVul, HSVulResponse, HSVulRequest
from std_srvs.srv import Trigger, TriggerResponse,TriggerRequest
import cv2 as cv
from cv_bridge import CvBridge

class redDetection:
    def __init__(self):
        ##ned to figure out how to pass the image message to threhold image
        self._sub = rospy.Subscriber('/kf/vehicle/oakd_bottom/color/image_raw',Image,self._handle_image)
        ##need to create a new topic to bpulish to
        self._pub = rospy.Publisher('color_filter/threshold_image',Image,queue_size=10)
        #self._srv1 = rospy.Service('color_filter/user_mask',HSVul,self._update_mask)
        #self._srv2 = rospy.Service('color_filter/reset_masks',Trigger,self._reset)
        self._bridge = CvBridge()
        self._mask1_lwr = (0,120,120)
        self._mask1_upr = (10, 255, 255)
        self._mask2_lwr = (165, 120, 120)
        self._mask2_upr = (179, 255, 255)

    
    
    ##use openCV to threshold image so only red color remains and everything else set to black
    def _handle_image(self,img_msg):
        
        ##check if this encoding is right
        img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        
        ##set everything to 255 (0xFFFFFF) the HSV values are in the correct range (hues above and below 0 are red)
        mask1 = cv.inRange(hsv_img,self._mask1_lwr,self._mask1_upr)
        mask2 = cv.inRange(hsv_img, self._mask2_lwr,self._mask2_upr)
        red_mask = cv.bitwise_or(mask1,mask2)
        ##do bitwise and to set everything that is not red to 0 and everything that is red to 1 
        img_red = cv.bitwise_and(hsv_img,hsv_img,mask=red_mask)
        final_img = cv.cvtColor(img_red,cv.COLOR_HSV2BGR)
        msg = self._bridge.cv2_to_imgmsg(final_img)
        self._pub.publish(msg)

    #def _apply_mask(self,HSVmsg: HSVulRequest) -> HSVulResponse:
        #response = HSVulResponse()
        #self._mask1_lwr = HSVmsg.mask1_lwr
        #self._mask1_upr = HSVmsg.mask1_upr
        #self._mask2_lwr = HSVmsg.mask2_lwr
        #self._mask2_upr = HSVmsg.mask2_upr
    

    #def _reset(self,msg:TriggerRequest) -> TriggerResponse:
        #self._mask1_lwr = (0,120,120)
        #self._mask1_upr = (10, 255, 255)
        #self._mask2_lwr = (165, 120, 120)
        #self._mask2_upr = (179, 255, 255)
        #r = TriggerResponse()
        
       
        
        

def main():
    rospy.init_node('red_detector')
    redDetection()
    rospy.spin()