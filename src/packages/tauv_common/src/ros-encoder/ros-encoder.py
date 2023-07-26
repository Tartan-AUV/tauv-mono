#!/usr/bin/env python
import rospy
import struct
import zlib
from std_msgs.msg import Float32, Float64, Int32, Int64

class BitstreamCompressor:
    def __init__(self):
        self.topics_to_subscribe = [
            ("/topic1", Float32),
            ("/topic2", Float64),
            ("/topic3", Int32),
            ("/topic4", Int64)
        ]
        self.message_compression_level = 0.5

        self.last_received_values = {topic: None for topic, _ in self.topics_to_subscribe}

        rospy.init_node('bitstream_compressor_node', anonymous=True)

        for topic, msg_type in self.topics_to_subscribe:
            rospy.Subscriber(topic, msg_type, self.callback, callback_args=topic)

    def compress_message(self, msg):
        serialized_msg = msg.serialize()  # Serialize the ROS message to a string
        compressed_msg = zlib.compress(serialized_msg, level=self.message_compression_level)

        return compressed_msg

    def callback(self, msg, topic):
        compressed_msg = self.compress_message(msg)

        bitstream = self.convert_to_bitstream(compressed_msg)

        # TODO: Transmit the bitstream over USB to the teensy4.1 (commented out for now)

    def convert_to_bitstream(self, compressed_msg):
        # Pack the bitstream length as a 4-byte unsigned integer
        bitstream_length = len(compressed_msg)
        bitstream_header = struct.pack("<I", bitstream_length)

        bitstream = bitstream_header + compressed_msg

        return bitstream

if __name__ == '__main__':
    try:
        compressor = BitstreamCompressor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
