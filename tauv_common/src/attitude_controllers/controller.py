# hybrid_controller.py
#
# This is an attitute
#
#

import rospy
import simple_pid
from dynamics.dynamics import Dynamics
from tauv_msgs.msg import ControllerCmd

class ControlAggregator:
    def __init__(self):
        self.dt = 0.02  # 50Hz
        self.dyn = Dynamics()
        self.target_acc = None
        self.target_yaw_acc = None
        self.target_roll = None
        self.target_pitch = None
        self.last_updated = None
        self.timeout_duration = 2.0  # timeout is 2 seconds. TODO: rosparam this

    def control_update(self):
        failsafe = False
        if self.last_updated is None or rospy.Time.now().to_sec() - self.last_updated > self.timeout_duration:
            failsafe = True
            return

        

    def plan_callback(self, msg):
        self.target_acc = [msg.a_x, msg.a_y, msg.a_z]
        self.target_yaw_acc = msg.a_yaw
        self.target_roll = msg.p_roll
        self.target_pitch = msg.p_pitch
        self.last_updated = rospy.Time.now().to_sec()

    def start(self):
        rospy.Timer(rospy.Duration.from_sec(self.dt), self.control_update)
