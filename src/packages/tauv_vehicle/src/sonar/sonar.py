from .ping_python.brping.ping360 import Ping360
from .ping_python.brping import definitions

from tauv_msgs.msg import SonarPulse
from std_msgs.msg import Header

import math

import rospy

RAD_PER_GRAD = math.pi / 200.0


class Sonar:
    def __init__(self):
        self.tty = rospy.get_param('/vehicle_params/sonar/tty')

        self.ping = Ping360(self.tty, baudrate=2000000)

        if not self.ping.initialize():
            raise ValueError("Unable to initialize ping360 device!")

        # self.ping.control_reset()

        self.gain_setting = rospy.get_param('/vehicle_params/sonar/gain')
        self.transmit_duration = rospy.get_param('/vehicle_params/sonar/transmit_duration')
        self.range = rospy.get_param('/vehicle_params/sonar/range')
        self.transmit_frequency = rospy.get_param('/vehicle_params/sonar/transmit_frequency')
        self.num_samples = rospy.get_param('/vehicle_params/sonar/num_samples')
        self.sonar_link = rospy.get_param('/vehicle_params/sonar/sonar_link', 'sonar_link')

        self.sample_period = self.calculateSamplePeriod(self.range, self.num_samples, 1500)

        self.pub_data = rospy.Publisher("data", SonarPulse, queue_size=10)

        self.updateSonarConfig()

        self.angular_speed_grad_per_s = 1000.0 * 400.0 / 2400.0
        self.angle = 0

        print("initialized sonar!")

    def update(self, timer_event):
        self.angle = (1 + self.angle) % 400
        self.do_pulse_at_angle(self.angle)
        self.poll_data()

    def poll_data(self):
        res = self.ping.wait_message([definitions.COMMON_NACK, definitions.PING360_DEVICE_DATA], 0.1)

        if res is None:
            return

        if res.message_id == definitions.PING360_DEVICE_DATA:
            self.pub_data.publish(self.create_sonar_msg(res))
        elif res.message_id == definitions.COMMON_NACK:
            rospy.logwarn("Error communicating with ping360 device:\n%s", res.nack_message)

    def do_pulse_at_angle(self, angle):
        self.ping.transmitAngleNoWait(angle)

    def calculateSamplePeriod(self, distance, numberOfSamples, speedOfSound, _samplePeriodTickDuration=25e-9):
        return 2 * distance / (numberOfSamples * speedOfSound * _samplePeriodTickDuration)

    def create_sonar_msg(self, pinger_data):
        msg = SonarPulse()
        msg.header = Header()
        msg.header.frame_id = self.sonar_link
        msg.header.stamp = rospy.Time.now()

        msg.mode = pinger_data.mode
        msg.gain_setting = pinger_data.gain_setting
        msg.angle = pinger_data.angle # * RAD_PER_GRAD
        msg.transmit_duration = pinger_data.transmit_duration
        msg.sample_period = pinger_data.sample_period
        msg.transmit_frequency = pinger_data.transmit_frequency
        msg.number_of_samples = pinger_data.number_of_samples
        msg.data_length = pinger_data.data_length
        msg.data = [b for b in bytearray(pinger_data.data)]
        return msg

    def updateSonarConfig(self):
        self.ping.set_gain_setting(self.gain_setting)
        self.ping.set_transmit_frequency(self.transmit_frequency)
        self.ping.set_transmit_duration(self.transmit_duration)
        self.ping.set_sample_period(self.sample_period)
        self.ping.set_number_of_samples(self.num_samples)

    def start(self):
        rospy.Timer(rospy.Duration(0.01), self.update)
        # rospy.Timer(rospy.Duration(0.01), self.poll_data)
        rospy.spin()


def main():
    rospy.init_node('sonar')
    s = Sonar()
    s.start()
