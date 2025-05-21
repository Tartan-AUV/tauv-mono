from arena_api.system import system
from arena_api.buffer import *

import ctypes
import numpy as np
import cv2
import time

import rclpy
from rclpy.node import Node 
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy   
from sensor_msgs.msg import Image

TAB1 = "  "
TAB2 = "    "

def create_devices_with_tries(logger):
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''

	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			logger.info(
				f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				print(f'{TAB1}{sec_count + 1 } seconds passed ',
					'.' * sec_count, end='\r')
			tries += 1
		else:
			logger.info(f'{TAB1}Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'{TAB1}No device found! Please connect a device and run '
						f'the example again.')



def setup(device):
    """
    Setup stream dimensions and stream nodemap
        num_channels changes based on the PixelFormat
        Mono 8 would has 1 channel, RGB8 has 3 channels

    """
    nodemap = device.nodemap
    nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat', 
                              'AcquisitionFrameRateEnable', 'AcquisitionFrameRate', 
                              'TransmissionFrameRate', 'AcquisitionMode', 'DeviceStreamChannelPacketSize',
                              'GainSelector', 'Gain', 'GainAuto', 'ExposureAuto', 'ExposureTime', 'GevSCPD', "DeviceLinkThroughputReserve","TCPEnable"]) # "ExposureTime"])
    if (not nodes['Width'].is_writable) : 
        raise Exception("FUCK")

    if (not nodes['AcquisitionFrameRateEnable'].is_writable): 
        raise Exception("FUCK")

    if (not nodes['DeviceStreamChannelPacketSize'].is_readable):
        raise Exception("asdf")
    else:
        print(nodes['DeviceStreamChannelPacketSize'].value)

    if (not nodes['DeviceStreamChannelPacketSize'].is_writable):
        raise Exception("asdf")
    else:
        nodes['DeviceStreamChannelPacketSize'].value = 9000
    nodes['Width'].value = 5320
    nodes['Height'].value = 3032
    nodes['PixelFormat'].value = 'RGB8'
    nodes['AcquisitionFrameRateEnable'].value = True
    nodes['AcquisitionFrameRate'].value = 1.0
    nodes['AcquisitionMode'].value = "Continuous"
    nodes['TCPEnable'].value = True
    nodes['GevSCPD'].value = 0
    nodes['DeviceLinkThroughputReserve'].value = 0
    nodes['ExposureAuto'].value = "Off"
    nodes['ExposureTime'].value = 20000.0
    print(nodes['GainAuto'])
    print(nodes['GainSelector'])
    nodes['Gain'] = 45.0

    num_channels = 3

    # Stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap

    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True

    return num_channels

def get_device(desired_ip):
    for device_info in system.device_infos:
        if device_info['ip'] == desired_ip:
            device= system.create_device([device_info])[0]
            return device

class LucidNode(Node):
    

    def __init__(self):
        super().__init__('lucid')
        self._bridge = CvBridge()
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self._image_pub = self.create_publisher(Image, '/image_raw', qos_profile=qos_profile)
        self._image_pub_small = self.create_publisher(Image, '/image_raw_small', qos_profile=qos_profile)
        self.get_logger().info("Lucid Node Initialized.")

    def callback(self):
        pass

    def start(self, device_ip):
        # devices = create_devices_with_tries(self.get_logger())
        self.get_logger().info(f'{system.device_infos}')
        device = get_device(device_ip)
        if device is None:
            raise Exception(f"Device with ip {device_ip} not found")
        
        num_channels = setup(device)

        curr_frame_time = 0
        prev_frame_time = 0

        with device.start_stream():

            while True:
                curr_frame_time = time.time()
                buffer = device.get_buffer()
                item = buffer
                buffer_bytes_per_pixel = 3
                array = (ctypes.c_ubyte * num_channels * item.width * item.height).from_address(ctypes.addressof(item.pbytes))
                cvframe = np.ndarray(buffer=array, dtype=np.uint8, shape=(item.height, item.width, buffer_bytes_per_pixel))

                fps = str(1 / (curr_frame_time - prev_frame_time))
                self.get_logger().info(f'FPS {fps}')

                try:
                    # Publish original image
                    img_msg = self._bridge.cv2_to_imgmsg(cvframe, encoding="rgb8")
                    self._image_pub.publish(img_msg)

                    # Downscale image
                    height, width = cvframe.shape[:2]
                    downscaled = cv2.resize(cvframe, (width // 20, height // 20), interpolation=cv2.INTER_AREA)
                    img_msg_small = self._bridge.cv2_to_imgmsg(downscaled, encoding="rgb8")
                    self._image_pub_small.publish(img_msg_small)

                    self.get_logger().info("Published original and downscaled images.")
                except CvBridgeError as e:
                    self.get_logger().error(f"Image conversion error: {e}")

                device.requeue_buffer(item)
                prev_frame_time = curr_frame_time

            device.stop_stream()
            cv2.destroyAllWindows()

        system.destroy_device()


def main(args=None):
    rclpy.init(args=args)
    node = LucidNode()
    device_ip = "10.0.2.11"
    node.start(device_ip)
    rclpy.spin=()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


