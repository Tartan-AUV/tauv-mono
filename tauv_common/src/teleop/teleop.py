# Teleop.py
# Converts joystick message into useful pose, twist, and accel messages
# Intended to be used as inputs to the pid_control_wrapper
# Usually you would only use a combination of the pose, twist, and accel outputs
# since they are separate results that represent different things.
# TODO: Send arm/disarm commands from the controller

import rospy

from geometry_msgs.msg import Pose, PoseStamped, Twist, Accel, Vector3, Point, Quaternion
from sensor_msgs.msg import Joy
import tf
from scipy.spatial import transform as stf
import math
from std_srvs.srv import SetBool


def build_cmd(joy, name):
    linear = Vector3(0, 0, 0)
    angular = Vector3(0, 0, 0)

    linear.x = rospy.get_param("~{}/axes/linear_x/scale".format(name)) * \
               joy.axes[rospy.get_param("~{}/axes/linear_x/axis".format(name))]

    linear.y = rospy.get_param("~{}/axes/linear_y/scale".format(name)) * \
               joy.axes[rospy.get_param("~{}/axes/linear_y/axis".format(name))]

    linear.z = rospy.get_param("~{}/axes/linear_z_down/scale".format(name)) * \
               joy.axes[rospy.get_param("~{}/axes/linear_z_down/axis".format(name))] \
               - rospy.get_param("~{}/axes/linear_z_up/scale".format(name)) * \
               joy.axes[rospy.get_param("~{}/axes/linear_z_up/axis".format(name))]

    angular.x = rospy.get_param("~{}/axes/angular_x/scale".format(name)) * \
                joy.axes[rospy.get_param("~{}/axes/angular_x/axis".format(name))]

    angular.y = rospy.get_param("~{}/axes/angular_y/scale".format(name)) * \
                joy.axes[rospy.get_param("~{}/axes/angular_y/axis".format(name))]

    angular.z = rospy.get_param("~{}/axes/angular_z/scale".format(name)) * \
                joy.axes[rospy.get_param("~{}/axes/angular_z/axis".format(name))]

    return linear, angular


class Teleop:
    def __init__(self):
        self.pub_cmd_pos = rospy.Publisher("cmd_pos", PoseStamped, queue_size=10)
        self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.pub_cmd_acc = rospy.Publisher("cmd_acc", Accel, queue_size=10)

        self.dt = 0.02
        self.pos = (0, 0, 0)
        self.orientation = (0, 0, 0, 1)
        self.joy = None

        self.tfl = tf.TransformListener()
        self.transformer = tf.Transformer()

        self.body = 'base_link'
        self.odom = 'odom'

        self.sub_joy = rospy.Subscriber(rospy.get_param("~joy_topic"), Joy, self.joy_callback)

    def joy_callback(self, joy):
        self.joy = joy

    def form_pose_message(self, linear, angular):
        # TODO: Support xy position and more depth/orientation modes
        # Depth is integrated, as is heading. Orientation and xy position are absolute.

        res = PoseStamped()
        res.header.stamp = rospy.Time.now()
        res.header.frame_id = self.odom
        depth = self.pos[2]
        res.pose.position.z = depth + self.dt * linear.z
        res.pose.position.x = linear.x
        res.pose.position.y = linear.y

        R = stf.Rotation.from_quat(self.orientation)
        ZYX = R.as_euler("ZYX")
        ZYX[0] += angular.z * self.dt
        ZYX[1] = angular.y
        ZYX[2] = angular.x

        if ZYX[0] > math.pi:
            ZYX[0] -= 2 * math.pi
        if ZYX[0] < -math.pi:
            ZYX[0] += 2 * math.pi

        quat = stf.Rotation.from_euler("ZYX", ZYX).as_quat()
        res.pose.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])

        return res

    def update(self, timer_event):
        if self.joy is None:
            return

        try:
            (self.pos, self.orientation) = self.tfl.lookupTransform(self.odom, self.body, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Failed to find transformation between frames {} and {}:\n {}".format(self.odom, self.body, e))
            return

        (pos_linear, pos_angular) = build_cmd(self.joy, "position")
        (vel_linear, vel_angular) = build_cmd(self.joy, "velocity")
        (acc_linear, acc_angular) = build_cmd(self.joy, "acceleration")

        cmd_vel = Twist()
        cmd_vel.angular = vel_angular
        cmd_vel.linear = vel_linear

        cmd_acc = Accel()
        cmd_acc.angular = acc_angular
        cmd_acc.linear = acc_linear

        cmd_pos = self.form_pose_message(pos_linear, pos_angular)

        self.pub_cmd_acc.publish(cmd_acc)
        self.pub_cmd_vel.publish(cmd_vel)
        self.pub_cmd_pos.publish(cmd_pos)

        if self.joy.buttons[rospy.get_param("~arm_button")] == 1:
            self.arm(True)

        if self.joy.buttons[rospy.get_param("~estop_button")] == 1:
            self.arm(False)

    def arm(self, arm):
        try:
            arm_srv = rospy.ServiceProxy('/arm', SetBool)
            resp1 = arm_srv(arm)
            return resp1.sum
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def start(self):
        rospy.Timer(rospy.Duration(self.dt), self.update)
        rospy.spin()


def main():
    rospy.init_node('teleop')
    t = Teleop()
    t.start()
