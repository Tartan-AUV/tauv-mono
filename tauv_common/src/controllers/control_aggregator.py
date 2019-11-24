# control_aggregator.py
#
# This node combines the outputs from all the controllers into one resultant wrench.
# The node will use the frame_id to transform the force/torque wrenches into the output frame.
# Currently it only supports rotation, and not translation of wrench frames. Support for translated
# torque wrenches is complex and possibly ambiguous, so it is TODO.
#
# This node requires parameters to be set in the "/(model_name)/controllers/configs" namespace.
# See the launchfile and the controllers.yaml in the vehicle_description for more info on parameterization.
#
# You can turn on and off controllers using the "controller_set_status" service.
#
# Author: Tom Scherlis 2019


import rospy
from geometry_msgs.msg import WrenchStamped, Wrench
from std_msgs.msg import Header
import tf
import numpy as np
from tauv_msgs.srv import SetController, SetControllerResponse
from tauv_msgs.msg import ControllerList, ControllerStatus


class ControlAggregator:
    def __init__(self):

        self.tf = tf.TransformListener()

        # retrieve parameters:
        self.controllerConfigs = rospy.get_param('configs/controllers')
        self.controllers = self.controllerConfigs.keys()
        self.topic_suffix = rospy.get_param('configs/topic_suffix')
        self.output_frame = rospy.get_param('configs/output_frame')
        self.timeout_s = rospy.get_param('configs/timeout_s')

        # controller statuses:

        self.lastTime = {c: None for c in self.controllers}
        self.lastWrench = {c: None for c in self.controllers}
        self.enable = {c: self.controllerConfigs[c]["enabled"] for c in self.controllers}
        self.frames = {c: "" for c in self.controllers}

        # subscribe to wrench topics:
        self.subscribers = []
        for c in self.controllers:
            sub = rospy.Subscriber(c + self.topic_suffix, WrenchStamped, self.wrenchCallback, callback_args=c)
            self.subscribers.append(sub)

        self.pub = rospy.Publisher(rospy.get_param('configs/output_topic'), WrenchStamped)
        self.statusPub = rospy.Publisher("status", ControllerList)

        self.enableSrv = rospy.Service('controller_set_status', SetController, self.setControllerCallback)

    def wrenchCallback(self, msg, controller):
        self.lastTime[controller] = msg.header.stamp
        self.frames[controller] = msg.header.frame_id
        self.lastWrench[controller] = msg.wrench

    def setControllerCallback(self, req):
        if req.name in self.controllers:
            self.enable[req.name] = req.enable
            res = SetControllerResponse()
            res.success = True
            return res
        else:
            res = SetControllerResponse()
            res.success = False
            res.message = "Controller does not exist!"
            return res

    def publishStatus(self, timerEvent):
        statuses = []
        for c in self.controllers:
            s = ControllerStatus()
            s.name = c
            s.enabled = self.enable[c]
            statuses.append(s)
        cl = ControllerList()
        cl.header = Header()
        cl.header.stamp = rospy.Time.now()
        cl.controllers = statuses
        self.statusPub.publish(cl)

    def publishWrench(self, timerEvent):
        resultant = Wrench()
        msg = WrenchStamped()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.output_frame

        for c in self.controllers:

            # verify that we have received a valid wrench within the timeout period:
            if not self.enable[c]:
                continue
            if self.lastWrench is None:
                continue
            if self.lastTime[c] is None or \
                    rospy.Time.now() - self.lastTime[c] > rospy.Duration(self.timeout_s):
                continue
            if self.frames[c] == "":
                continue

            # grab last wrench in controller frame:
            wrench_controller = self.lastWrench[c]
            f_c = wrench_controller.force
            f_c = np.array([f_c.x, f_c.y, f_c.z])

            t_c = wrench_controller.torque
            t_c = np.array([t_c.x, t_c.y, t_c.z])

            # transform the wrench to the base_link:
            try:
                (trans, rot) = self.tf.lookupTransform(self.output_frame, self.frames[c], rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                # TODO raise tauv_exception here!
                print("Failed to find transformation between frames: {}".format(e))
                continue

            # currently only support frames with a shared origin! throw out trans.
            H = tf.transformations.quaternion_matrix(rot)
            R = H[0:3, 0:3]
            f_b = np.dot(R, f_c)
            t_b = np.dot(R, t_c)

            resultant.force.x += f_b[0]
            resultant.force.y += f_b[1]
            resultant.force.z += f_b[2]
            resultant.torque.x += t_b[0]
            resultant.torque.y += t_b[1]
            resultant.torque.z += t_b[2]

        msg.wrench = resultant
        self.pub.publish(msg)

    def start(self):
        print(("Starting control aggregator with params:\n"
               "controllers: {}\n"
               "enabled: {}\n"
               "topic_suffix: {}\n"
               "output_frame: {}\n"
               "timeout: {}s").format(
            self.controllers,
            self.enable,
            self.topic_suffix,
            self.output_frame,
            self.timeout_s))

        # Start publishers at 100Hz and 2Hz respectively:
        rospy.Timer(rospy.Duration(1.0 / 100), self.publishWrench)
        rospy.Timer(rospy.Duration(1.0 / 2), self.publishStatus)
        rospy.spin()


def main():
    rospy.init_node('control_aggregator')
    ca = ControlAggregator()
    ca.start()
