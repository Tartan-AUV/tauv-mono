import rospy
from tauv_msgs.msg import FluidDepth as DepthMsg
from tauv_alarms import Alarm, AlarmClient
import serial


class Arduino:

    def __init__(self):
        self._ac: AlarmClient = AlarmClient()

        self._dt: float = 0.10

        self._depth_pub: rospy.Publisher = rospy.Publisher('depth', DepthMsg, queue_size=10)

        self._serial = None

        while not self._init() and not rospy.is_shutdown():
            rospy.loginfo('init failed, retrying')
            rospy.sleep(1)

        if rospy.is_shutdown():
            return

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _init(self) -> bool:
        try:
            self._serial = serial.Serial('/dev/arduino', 115200, timeout=0.05)  # try to establish serial connection
            return True
        except (IOError, OSError, serial.SerialException) as e:
            rospy.logerr(f'_init: {e}')
            return False

    def _update(self, timer_event):
        try:
            serial_data = self._serial.readline().decode("UTF-8").strip()
        except (IOError, OSError) as e:
            rospy.logerr(f'_update: {e}')
            while not self._init() and not rospy.is_shutdown():
                rospy.loginfo('init failed, retrying')
                rospy.sleep(1)
            return

        serial_data_split = serial_data.split(",")
        serial_message_type = serial_data_split[0]

        timestamp = rospy.Time.now()

        # rospy.loginfo(serial_data)

        if serial_data_split[0] == "D":
            if serial_data_split[1].lower() != "nan":
                depth_msg = DepthMsg()
                depth_msg.header.stamp = timestamp
                depth_msg.header.frame_id = 'depth_sensor_link'
                depth_msg.depth = float(serial_data_split[1])

                self._depth_pub.publish(depth_msg)
            else:
                rospy.logwarn("bad depth reading")
        else:
            pass
            # rospy.logwarn(f'unknown message type: {serial_message_type}')

        self._ac.clear(Alarm.ARDUINO_NOT_INITIALIZED)


def main():
    rospy.init_node('arduino')
    a = Arduino()
    a.start()