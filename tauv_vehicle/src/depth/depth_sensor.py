import rospy
from sensor_msgs.msg import Temperature
from std_msgs.msg import Float64
from ms5837-python import ms5837

pub = rospy.Publisher('depth', Float64, queue_size=10)
pub = rospy.Publisher('temperature', Temperature, queue_size=10)
rospy.init_node('depth_sensor')



class depth_sensor():
   def __init__(self):
      self.pub_depth = rospy.Publisher('depth', Float64, queue_size=10)
      self.pub_temp = rospy.Publisher('temperature', Temperature, queue_size=10)
      self.ms5837 = ms5837.MS5837_30BA()

   def start(self):
      r = rospy.Rate(10) # 10hz
      while not rospy.is_shutdown():
         self.ms5837.read()
         self.pub_depth.publish(Float64(self.ms5837.depth()))
         self.pub_temp.publish(Temperature(self.ms5837.temperature(), 0))
         r.sleep()


if __name__ == '__main__':
   d = depth_sensor()
   rospy.init_node('depth_sensor')
   d.start()