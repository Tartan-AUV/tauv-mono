import rospy
import depthai
import cv2 as cv
from std_msgs import Image

NODENAME = "oakd_publisher"
QUEUE_SIZE = 100
# node to create oakd ros nodes from oakd api
# publishes depth map and color image

class OAKDNode:
    def __init__(self):

        rospy.init(NODENAME, anonymous = True)

        self.pipeline = depthai.Pipeline()

        self.cam_rgb = self.pipeline.create(depthai.node.ColorCamera)
        self.stereo = self.pipeline.create(depthai.node.StereoDepth)
        self.left = self.pipeline.create(depthai.node.MonoCamera)
        self.right = self.pipeline.create(depthai.node.MonoCamera)

        self.left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        self.right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

        self.xout_color = self.pipeline.create(depthai.node.XLinkOut)
        self.xout_depth = self.pipeline.create(depthai.node.XLinkOut)

        self.xout_color.setStreamName("rgb")
        self.xout_depth.setStreamName("depth")

        self.left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.left.setBoardSocket(depthai.CameraBoardSocket.LEFT)

        self.right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
        self.right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

        self.cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
        self.cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam_rgb.setInterleaved(False)
        self.cam_rgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)
        
        self.stereo.setLeftRightCheck(False)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)
        self.stereo.setDepthAlign(depthai.CameraBoardSocket.RGB)
        self.stereo.setDefaultProfilePreset(depthai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        self.stereo.initialConfig.setMedianFilter(depthai.MedianFilter.KERNEL_7x7)

        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)
        self.stereo.depth.link(self.xout_depth.input)
        self.cam_rgb.video.link(self.xout_color.input)

        self.depth = rospy.Publisher("/oakd/oakd_front/depth_map", Image, queue_size=10)
        self.color = rospy.Publisher("/oakd/oakd_front/color_image", Image, queue_size=10)
        self.camera_info = rospy.Service("/oakd/oakd_front/camera_info", )


        with depthai.Device(self.pipeline) as device:
            device.startPipeline(self.pipeline)

            # Create a receive queue for each stream
            qList = [device.getOutputQueue(stream, QUEUE_SIZE, blocking=False) for stream in streams]

            while True:
            # Output queues will be used to get the grayscale frames from the outputs defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)


            while True:
                # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
                inLeft = qLeft.tryGet()
                inRight = qRight.tryGet()

                if inLeft is not None:
                    nLeft.getCvFrame()

                if inRight is not None:
                    cv.imshow("right", inRight.getCvFrame())


# frame id of header oakd_front

def main():
    OAKDNode()
