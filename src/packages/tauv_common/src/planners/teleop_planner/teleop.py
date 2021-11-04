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
from tauv_msgs.msg import ControllerCmd


def build_cmd(joy, orientation):
    linear = Vector3(0, 0, 0)
    angular = Vector3(0, 0, 0)

    x = rospy.get_param("~axes/linear_x/scale") * \
               joy.axes[rospy.get_param("~axes/linear_x/axis")]

    y = rospy.get_param("~axes/linear_y/scale") * \
               joy.axes[rospy.get_param("~axes/linear_y/axis")]

    z = rospy.get_param("~axes/linear_z_down/scale") * \
               joy.axes[rospy.get_param("~axes/linear_z_down/axis")] \
               - rospy.get_param("~axes/linear_z_up/scale") * \
               joy.axes[rospy.get_param("~axes/linear_z_up/axis")]

    yaw = stf.Rotation.from_quat(orientation).as_euler('ZYX')[0]
    R = stf.Rotation.from_euler('ZYX', [yaw, 0, 0])
    xyz_world = R.apply([x,y,z])

    linear.x = xyz_world[0]
    linear.y = xyz_world[1]
    linear.z = xyz_world[2]

    angular.x = rospy.get_param("~axes/angular_x/scale") * \
                joy.axes[rospy.get_param("~axes/angular_x/axis")]

    angular.y = rospy.get_param("~axes/angular_y/scale") * \
                joy.axes[rospy.get_param("~axes/angular_y/axis")]

    angular.z = rospy.get_param("~axes/angular_z/scale") * \
                joy.axes[rospy.get_param("~axes/angular_z/axis")]

    return linear, angular


class Teleop:
    def __init__(self):
        self.pub_cmd = rospy.Publisher("controller_cmd", ControllerCmd, queue_size=10)

        self.dt = 0.02
        self.pos = (0, 0, 0)
        self.orientation = (0, 0, 0, 1)
        self.joy = None

        self.tfl = tf.TransformListener()
        self.transformer = tf.Transformer()

        self.body = 'base_link'
        self.odom = 'odom'

        self.sub_joy = rospy.Subscriber('joy', Joy, self.joy_callback)

    def joy_callback(self, joy):
        self.joy = joy

    def update(self, timer_event):
        if self.joy is None:
            rospy.logwarn_throttle(3, "No joystick data received yet! Teleop node waiting.")
            return

        try:
            (self.pos, self.orientation) = self.tfl.lookupTransform(self.odom, self.body, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Failed to find transformation between frames {} and {}:\n {}".format(self.odom, self.body, e))
            return

        (cmd_linear, cmd_angular) = build_cmd(self.joy, self.orientation)

        cmd = ControllerCmd()
        cmd.a_x = cmd_linear.x
        cmd.a_y = cmd_linear.y
        cmd.a_z = cmd_linear.z
        cmd.a_yaw = cmd_angular.z
        cmd.p_roll = cmd_angular.x
        cmd.p_pitch = cmd_angular.y
        self.pub_cmd.publish(cmd)

        if self.joy.buttons[rospy.get_param("~arm_button")] == 1:
            self.arm(True)

        if self.joy.buttons[rospy.get_param("~estop_button")] == 1:
            self.arm(False)

    def arm(self, arm):
        try:
            arm_srv = rospy.ServiceProxy('/arm', SetBool)
            resp1 = arm_srv(arm)
            return resp1.success
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(3, "[Teleop Planner] Arm server not responding, cannot arm/disarm robot.")

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self.dt), self.update)
        rospy.spin()


def main():
    rospy.init_node('teleop')
    t = Teleop()
    t.start()

