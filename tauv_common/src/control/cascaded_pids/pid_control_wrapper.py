# Wraps the cascaded pid controllers and allows input selections

import rospy

from tauv_msgs.msg import CascadedPidSelection
from tauv_msgs.srv import SetCascadedPidSelection, SetCascadedPidSelectionResponse


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

        self.sub_joy_pos = rospy.Subscriber("joy_cmd_pos", self.callback_cmd_pos, )
        self.sub_joy_vel = rospy.Subscriber("joy_cmd_vel")
        self.sub_joy_acc = rospy.Subscriber("joy_cmd_acc")
        self.sub_control_pos = rospy.Subscriber("controller_cmd_pos")
        self.sub_control_vel = rospy.Subscriber("controller_cmd_vel")
        self.sub_control_acc = rospy.Subscriber("controller_cmd_acc")
        self.pub_cmd_pos = rospy.publish

    def loadDefaultConfig(self):
        self.selections.enableBuoyancyComp = rospy.get_param("~enableBuoyancyComp")
        self.selections.enableVelocityFeedForward = rospy.get_param("~enableVelocityFeedForward")
        self.selections.pos_src_heading = parse(rospy.get_param("~pos_src_heading"))
        self.selections.pos_src_attitude = parse(rospy.get_param("~pos_src_attitude"))
        self.selections.vel_src_translation = parse(rospy.get_param("~vel_src_translation"))
        self.selections.vel_src_heading = parse(rospy.get_param("~vel_src_heading"))
        self.selections.vel_src_attitude = parse(rospy.get_param("~vel_src_attitude"))
        self.selections.acc_src_translation = parse(rospy.get_param("~acc_src_translation"))
        self.selections.acc_src_heading = parse(rospy.get_param("~acc_src_heading"))
        self.selections.acc_src_attitude = parse(rospy.get_param("~acc_src_attitude"))


def main():
    rospy.init_node('pid_control_wrapper')
    pcw = PidControlWrapper()
    rospy.spin()
