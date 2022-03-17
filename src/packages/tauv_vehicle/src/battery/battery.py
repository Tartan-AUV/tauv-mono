import rospy
from smbus2 import SMBus

from tauv_msgs.msg import Battery as BatteryMsg


class Battery:

    def __init__(self):
        self._dt: float = 1.0

        self._load_config()

        self._bus: SMBus = SMBus(self._bus_index)

        self._battery_pub: rospy.Publisher = rospy.Publisher('battery', BatteryMsg, queue_size=10)

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        state_of_charge: float = self._bus.read_word_data(self._addr, 0x02) * 1e0
        voltage: float = self._bus.read_word_data(self._addr, 0x08) * 1e-3
        average_current: float = self._bus.read_word_data(self._addr, 0x0A) * 1e-3
        remaining_capacity: float = self._bus.read_word_data(self._addr, 0x04) * 1e0
        full_capacity: float = self._bus.read_word_data(self._addr, 0x06) * 1e0

        msg: BatteryMsg = BatteryMsg()
        msg.header.stamp = rospy.Time.now()
        msg.state_of_charge = state_of_charge
        msg.voltage = voltage
        msg.average_current = average_current
        msg.remaining_capacity = remaining_capacity
        msg.full_capacity = full_capacity
        self._battery_pub.publish(msg)

    def _load_config(self):
        self._bus_index = rospy.get_param('~bus_index')
        self._addr = rospy.get_param('~address')


def main():
    rospy.init_node('battery')
    b = Battery()
    b.start()
