import rospy

from geometry_msgs.msg import TwistWithCovarianceStamped, Vector3
from tauv_msgs.msg import TeledyneDvlData as TeledyneDvlMsg


class TeledyneDvl:

    def __init__(self):
        self._sim_dvl_sub: rospy.Subscriber = rospy.Subscriber('sim_dvl', TwistWithCovarianceStamped, self._handle_sim_dvl)
        self._dvl_pub: rospy.Publisher = rospy.Publisher('dvl', TeledyneDvlMsg, queue_size=10)

    def start(self):
        rospy.spin()

    def _handle_sim_dvl(self, msg: TwistWithCovarianceStamped):
        m = TeledyneDvlMsg()
        m.header.stamp = rospy.Time.now()
        m.is_hr_velocity_valid = True
        m.hr_velocity = Vector3(msg.twist.twist.linear.y, msg.twist.twist.linear.x, -msg.twist.twist.linear.z)
        self._dvl_pub.publish(m)

def main():
    rospy.init_node('teledyne_dvl')
    t = TeledyneDvl()
    t.start()