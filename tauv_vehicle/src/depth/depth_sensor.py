#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Temperature
from std_msgs.msg import Float64
from std_msgs.msg import Header
from ms5837lib import ms5837
from time import sleep

pub = rospy.Publisher('depth', Float64, queue_size=10)
pub = rospy.Publisher('temperature', Temperature, queue_size=10)
rospy.init_node('depth_sensor')



class depth_sensor():
   def __init__(self):
      self.pub_depth = rospy.Publisher('depth', Float64, queue_size=10)
      self.pub_temp = rospy.Publisher('temperature', Temperature, queue_size=10)
      self.ms5837 = ms5837.MS5837_30BA()
      while not self.ms5837.init():
          rospy.sleep(10)
          print("Failed to initialize depth sensor, retrying in 10 seconds")
      print("Depth sensor initialized!")

   def start(self):
      r = rospy.Rate(10) # 10hz
      while not rospy.is_shutdown():
         self.ms5837.read()
         self.pub_depth.publish(Float64(self.ms5837.depth()))
         tempmsg = Temperature()
         tempmsg.header = Header()
         tempmsg.header.stamp = rospy.Time.now()
         tempmsg.temperature = self.ms5837.temperature()
         self.pub_temp.publish(tempmsg)
         r.sleep()

def main():
   d = depth_sensor()
   rospy.init_node('depth_sensor')
   d.start()

if __name__ == '__main__':
    main()
