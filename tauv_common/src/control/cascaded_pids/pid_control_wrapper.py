# Wraps the cascaded pid controllers and allows input selections

import rospy

from tauv_msgs.msg import CascadedPidSelection
from tauv_msgs.srv import SetCascadedPidSelection, SetCascadedPidSelectionResponse
from geometry_msgs.msg import Pose, Twist, Accel

SOURCE_JOY = 1
SOURCE_CONTROLLER = 2


def parse(str):
    if str == "controller":
        return CascadedPidSelection.CONTROLLER
    elif str == "joy":
        return CascadedPidSelection.JOY
    raise ValueError("YAML Selections must be \"controller\" or \"joy\"")


class PidControlWrapper:
    def __init__(self):
        self.selections = CascadedPidSelection()
        self.loadDefaultConfig()

        self.sub_joy_pos = rospy.Subscriber("joy_cmd_pos", Pose, self.callback_cmd_pos,
                                            callback_args=SOURCE_JOY)
        self.sub_joy_vel = rospy.Subscriber("joy_cmd_vel", Twist, self.callback_cmd_vel,
                                            callback_args=SOURCE_JOY)
        self.sub_joy_acc = rospy.Subscriber("joy_cmd_acc", Accel, self.callback_cmd_acc,
                                            callback_args=SOURCE_JOY)
        self.sub_control_pos = rospy.Subscriber("controller_cmd_pos", Pose, self.callback_cmd_pos,
                                                callback_args=SOURCE_CONTROLLER)
        self.sub_control_vel = rospy.Subscriber("controller_cmd_vel", Twist, self.callback_cmd_vel,
                                                callback_args=SOURCE_CONTROLLER)
        self.sub_control_acc = rospy.Subscriber("controller_cmd_acc", Accel, self.callback_cmd_acc,
                                                callback_args=SOURCE_CONTROLLER)
        self.pub_cmd_pos = rospy.Publisher("cmd_pos", Pose, queue_size=10)
        self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.pub_cmd_acc = rospy.Publisher("cmd_acc", Accel, queue_size=10)
        self.pub_status = rospy.Publisher("status", CascadedPidSelection, queue_size=10)

        # State vars:
        self.joy_acc = None
        self.control_acc = None
        self.joy_vel = None
        self.control_vel = None
        self.joy_pos = None
        self.control_pos = None

    def loadDefaultConfig(self):
        self.selections.enableBuoyancyComp = rospy.get_param("~enableBuoyancyComp")
        self.selections.enableVelocityFeedForward = rospy.get_param("~enableVelocityFeedForward")
        self.selections.pos_src_z = parse(rospy.get_param("~pos_src_z"))
        self.selections.pos_src_heading = parse(rospy.get_param("~pos_src_heading"))
        self.selections.pos_src_attitude = parse(rospy.get_param("~pos_src_attitude"))
        self.selections.vel_src_xy = parse(rospy.get_param("~vel_src_xy"))
        self.selections.vel_src_z = parse(rospy.get_param("~vel_src_z"))
        self.selections.vel_src_heading = parse(rospy.get_param("~vel_src_heading"))
        self.selections.vel_src_attitude = parse(rospy.get_param("~vel_src_attitude"))
        self.selections.acc_src_xy = parse(rospy.get_param("~acc_src_xy"))
        self.selections.acc_src_z = parse(rospy.get_param("~acc_src_z"))
        self.selections.acc_src_heading = parse(rospy.get_param("~acc_src_heading"))
        self.selections.acc_src_attitude = parse(rospy.get_param("~acc_src_attitude"))

    def callback_cmd_acc(self, acc, source):
        # Acceleration is in the body frame!
        if source == SOURCE_JOY:
            self.joy_acc = acc
        elif source == SOURCE_CONTROLLER:
            self.control_acc = acc

    def callback_cmd_vel(self, vel, source):
        # Velocity is in the body frame!
        if source == SOURCE_JOY:
            self.joy_vel = vel
        elif source == SOURCE_CONTROLLER:
            self.control_vel = vel

    def callback_cmd_pos(self, pos, source):
        # Pose is in the world frame!
        if source == SOURCE_JOY:
            self.joy_pos = pos
        elif source == SOURCE_CONTROLLER:
            self.control_pos = pos

    def update(self, timer_event):
        self.pub_cmd_pos.publish(self.calculate_pos())
        self.pub_cmd_vel.publish(self.calculate_vel())
        self.pub_cmd_acc.publish(self.calculate_acc())

    def post_status(self, timer_event):
        self.pub_status.publish(self.selections)

    def start(self):
        rospy.Timer(rospy.duration(1 / self.frequency), self.update)
        rospy.Timer(rospy.duration(0.5), self.post_status)
        rospy.spin()



def main():
    rospy.init_node('pid_control_wrapper')
    pcw = PidControlWrapper()
    pcw.start()
