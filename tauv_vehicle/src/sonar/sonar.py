from ping_python.brping.ping360 import Ping360
from ping_python.brping import definitions

from tauv_msgs.msg import SonarPulse
from std_msgs.msg import Header

import math

import rospy

RAD_PER_GRAD = math.pi / 200.0

class Sonar:
    def __init__(self):
        self.tty = rospy.get_param('/vehicle_params/pinger/tty')

        self.ping = Ping360(self.tty)

        if not self.ping.initialize():
            raise ValueError("Unable to initialize ping360 device!")

        self.ping.control_reset()

        self.gain_setting = rospy.get_param('/vehicle_params/pinger/gain')
        self.transmit_duration = rospy.get_param('/vehicle_params/pinger/transmit_duration')
        self.sample_period = rospy.get_param('/vehicle_params/pinger/sample_period')
        self.transmit_frequency = rospy.get_param('/vehicle_params/transmit_frequency')
        self.sample_frequency = rospy.get_param('/vehicle_params/pinger/sample_frequency')
        self.num_samples = rospy.get_param('/vehicle_params/pinger/num_samples')
        self.sonar_link = rospy.get_param('/vehicle_params/pinger/sonar_link', 'sonar_link')

        self.updateSonarConfig()

        self.angular_speed_grad_per_s = 1000.0 * 400.0 / 2400.0
        self.angle = 0

    def do_pulse_at_angle(self, angle):
        res = self.ping.transmitAngle(angle)
        if res.message_id == definitions.PING360_DEVICE_DATA:
            self.create_sonar_msg(res)
        elif res.message_id == definitions.COMMON_NACK:
            rospy.logwarn("Error communicating with ping360 device:\n%s",res.nack_message)

    def create_sonar_msg(self, pinger_data):
        msg = SonarPulse()
        msg.header = Header()
        msg.header.frame_id = self.sonar_link
        msg.header.timestamp = rospy.Time.now()

        msg.mode = pinger_data.mode
        msg.gain_setting = pinger_data.gain_setting
        msg.angle = pinger_data.angle * RAD_PER_GRAD
        msg.transmit_duration = pinger_data.transmit_duration / 1000000.0
        msg.sample_period = pinger_data.sample_period * 25.0 / 1000000000.0
        msg.transmit_frequency = pinger_data.transmit_frequency * 1000.0
        msg.number_of_samples = pinger_data.number_of_samples
        msg.data_length = pinger_data.data_length
        msg.data = [b for b in bytearray(pinger_data.data)]

    def updateSonarConfig(self):
        self.ping.set_gain_setting(self.gain_setting)
        self.ping.set_transmit_frequency(self.transmit_frequency)
        self.ping.set_transmit_duration(self.transmit_duration)
        self.ping.set_sample_period(self.sample_period)
        self.ping.set_number_of_samples(self.num_samples)

