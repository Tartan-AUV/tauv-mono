import rospy
from sensor_msgs.msg import Temperature as TemperatureMsg
from tauv_msgs.msg import FluidDepth as DepthMsg
from ms5837lib import ms5837


class DepthSensor():
    def __init__(self):
        self._dt = 0.10
        self._depth_pub = rospy.Publisher('depth', DepthMsg, queue_size=10)
        self._temp_pub = rospy.Publisher('temperature', TemperatureMsg, queue_size=10)

        self._ms5837 = ms5837.MS5837_02BA()

        while not self._ms5837.init() and not rospy.is_shutdown():
            rospy.sleep(1)
            print("Failed to initialize depth sensor, retrying in 3 seconds")

        if rospy.is_shutdown():
            return

        print("Depth sensor initialized!")

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self._dt), self._update)
        rospy.spin()

    def _update(self, timer_event):
        self._ms5837.read()

        timestamp = rospy.Time.now()

        depth_msg = DepthMsg()
        depth_msg.header.stamp = timestamp
        depth_msg.header.frame_id = 'depth_sensor_link'
        depth_msg.depth = self._ms5837.depth()
        self._depth_pub.publish(depth_msg)

        temp_msg = TemperatureMsg()
        temp_msg.header.stamp = timestamp
        temp_msg.temperature = self._ms5837.temperature()
        self._temp_pub.publish(temp_msg)


def main():
    rospy.init_node('depth_sensor')
    d = DepthSensor()
    d.start()

