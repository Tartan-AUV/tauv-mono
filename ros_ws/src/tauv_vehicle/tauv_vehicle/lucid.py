from arena_api.system import system
from arena_api.buffer import *
import arena_api

import ctypes
import numpy as np
import cv2
import time

import rclpy
from rclpy.node import Node 
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy   
from sensor_msgs.msg import Image

import vpi

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



def setup(device, binning_dims):
    """
    Setup stream dimensions and stream nodemap
        num_channels changes based on the PixelFormat
        Mono 8 would has 1 channel, RGB8 has 3 channels

        device: the device (camera in this context)
        binning_dims: the dimensions (height, width) of binning 

    """
    nodemap = device.nodemap
    print(nodemap)
    nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat', 
                              'AcquisitionFrameRateEnable', 'AcquisitionFrameRate', 
                              'TransmissionFrameRate', 'AcquisitionMode', 'DeviceStreamChannelPacketSize',
                              'BinningHorizontal', 'BinningVertical', 'BinningSelector',
                              'BinningHorizontalMode', 'BinningVerticalMode', 'TransportStreamProtocol',
                              'TriggerNode', 'LineMode', 'TriggerMode', 'TriggerSelector', 'TriggerSource',
                              'TriggerActivation', 'OffsetX', 'OffsetY'])

    # Stopping image stream
    device.stop_stream()
 
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
        # nodes['DeviceStreamChannelPacketSize'].value = 8192
        nodes['DeviceStreamChannelPacketSize'].value = 9000
        

    # nodes['Height'].value = 758
    # nodes['Width'].value = 1328

    

    nodes['BinningSelector'].value = 'Digital'
    nodes['BinningHorizontalMode'].value = 'Sum'
    nodes['BinningVerticalMode'].value = 'Sum'

    if (not nodes['BinningVertical'].is_writable):
        raise Exception("Binning Vertical Not Writable")
    else:
        nodes['BinningVertical'].value = 1 #binning_dims[0]
        print("Did binning")
    if (not nodes['BinningHorizontal'].is_writable):
        raise Exception("Binning Horizontal Not Writable")
    else:
        nodes['BinningHorizontal'].value = 1 #binning_dims[1]
        print("Did binning")

    print(nodes['PixelFormat'])
    print(nodes['BinningVertical'])


    # Image Offset
    if (not nodes['OffsetX'].is_writable):
        raise Exception("OffsetX not writable")
    if (not nodes['OffsetY'].is_writable):
        raise Exception("OffsetY not writable")

    nodes['OffsetX'] = 0
    nodes['OffsetY'] = 0

    # Image Resolution
    nodes['Width'].value = 5320
    nodes['Height'].value = 3032



    nodes['PixelFormat'].value = 'RGB8'
    nodes['AcquisitionFrameRateEnable'].value = True
    nodes['AcquisitionFrameRate'].value = 10.0
    # nodes['source /opt/ros/humble/setup.bash'].value = 10.0
    nodes['AcquisitionMode'].value = "Continuous"

    nodes['TransportStreamProtocol'].value = "TCP"
    # print(nodes['GigEVision'])
    # print(nodes['GigEVision'].is_writable)

    num_channels = 3

    # Stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap

    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True


    # Pulsing for Image Collection
    if (not nodes['LineSelector'].is_writable):
        raise Exception("LineSelector not writable")
    if (not nodes['LineMode'].is_writable):
        raise Exception("LineMode not writable")
    if (not nodes['TriggerMode'].is_writable):
        raise Exception("TriggerMode not writable")
    if (not nodes['TriggerSelector'].is_writable):
        raise Exception("TriggerSelector not writable")
    if (not nodes['TriggerSource'].is_writable):
        raise Exception("TriggerSource not writable")
    if (not nodes['TriggerActivation'].is_writable):
        raise Exception("TriggerActivation not writable")

    nodes['LineSelector'] = 'Line0'
    nodes['LineMode'] = 'Input'

    nodes['TriggerMode'] = 'On'
    nodes['TriggerSelector'] = 'FrameStart'
    nodes['TriggerSource'] = 'Line0'
    nodes['TriggerActivation'] = 'FallingEdge'
    
    # Restart Stream
    #device.start_stream()

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
        
        self.declare_parameter('camera_ip', '10.0.2.11')
        self.declare_parameter('topic_name', '/image_raw')
        
        self.declare_parameter('horizontal_binning', 2)
        self.declare_parameter('vertical_binning', 2)


        self.camera_ip = self.get_parameter('camera_ip').get_parameter_value().string_value
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        self.horizontal_binning = self.get_parameter('horizontal_binning').get_parameter_value().integer_value
        self.vertical_binning = self.get_parameter('vertical_binning').get_parameter_value().integer_value

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        self._image_pub = self.create_publisher(Image, self.topic_name, qos_profile=qos_profile,)
        self.get_logger().info(f"Lucid Node Initialized with Camera IP: {self.camera_ip} and Topic: {self.topic_name}")


    def callback(self):
        pass
        


    def start(self):
        # devices = create_devices_with_tries(self.get_logger())

        backend = vpi.Backend.VIC

        device_ip = self.camera_ip
        device = get_device(device_ip)
        if device is None:
            raise Exception(f"Device with ip {device_ip} not found")

        # binning dimensionis (height, width)
        binning_dims = (self.vertical_binning, self.horizontal_binning)
        
        num_channels = setup(device, binning_dims)

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

                vpi_frame = vpi.asimage(cvframe)
                
                with backend:
                    temp = vpi_frame.convert(vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)
                    temp = temp.rescale((vpi_frame.width//int(self.horizontal_binning), vpi_frame.height//int(self.vertical_binning)))
                    output = temp.convert(vpi.Format.RGB8, backend=vpi.Backend.CUDA)
                
                output_frame = output.cpu()

                fps = str(1/(curr_frame_time - prev_frame_time))
                self.get_logger().info(f'FPS {fps}')
                adj_width = output_frame.shape[1]
                adj_height = output_frame.shape[0]
                self.get_logger().info(f'Image Size ({adj_width},{adj_height})')
                

                # # ------------------------------------
               
                # # Publish image
                try:
                    img_msg = self._bridge.cv2_to_imgmsg(output_frame, encoding="rgb8")
                    self._image_pub.publish(img_msg)
                    self.get_logger().info("Published Image :D")
                except CvBridgeError as e:
                    self.get_logger().info(f"Lucid Frame Error: {e}")
                # Destory copied item to prevent memory leak
                device.requeue_buffer(item)
                prev_frame_time = curr_frame_time
                

            device.stop_stream()
            cv2.destroyAllWindows()

        system.destroy_device()


def main(args=None):
    rclpy.init(args=args)
    node = LucidNode()
    node.start()
    rclpy.spin()
    # TODO: FIX THIS

if __name__ == "__main__":
    main()


