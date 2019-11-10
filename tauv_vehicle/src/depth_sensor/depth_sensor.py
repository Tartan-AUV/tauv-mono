#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Temperature
from std_msgs.msg import Header
from tauv_msgs.msg import FluidDepth
from ms5837lib import ms5837

class DepthSensor():
    def __init__(self):
        if not rospy.get_param('/vehicle_params/has_depth_sensor'):
            raise ValueError('''Error: Vehicle does not support depth sensor.
         Is the has_depth_sensor rosparam set in the vehicle_params.yaml?
         If not, then don't launch this node! ''')

        self.pub_depth = rospy.Publisher('depth', FluidDepth, queue_size=10)
        self.pub_temp = rospy.Publisher('temperature', Temperature, queue_size=10)

        model_name = rospy.get_param('/vehicle_params/depth_sensor')
        if model_name == 'bar30':
            self.ms5837 = ms5837.MS5837_30BA()
        elif model_name == 'bar02':
            self.ms5837 = ms5837.MS5837_02BA()
        else:
            raise ValueError('''Error: specified depth sensor not supported.
         Supported depth sensors are \'bar30\' and \'bar02\', set with the
         depth_sensor tag in the vehicle_params.yaml.''')

        while not self.ms5837.init() and not rospy.is_shutdown():
            rospy.sleep(3)
            print("Failed to initialize depth sensor, retrying in 3 seconds")
        if rospy.is_shutdown():
            return
        print("Depth sensor initialized!")

    def start(self):
        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            self.ms5837.read()

            tempmsg = Temperature()
            tempmsg.header = Header()
            tempmsg.header.stamp = rospy.Time.now()
            tempmsg.temperature = self.ms5837.temperature()
            self.pub_temp.publish(tempmsg)

            depthmsg = FluidDepth()
            depthmsg.header = Header()
            depthmsg.header.stamp = rospy.Time.now()
            depthmsg.header.frame_id = "odom"
            depthmsg.depth = self.ms5837.depth()
            self.pub_depth.publish(depthmsg)
            r.sleep()


def main():
    rospy.init_node('depth_sensor')
    d = DepthSensor()
    d.start()

