#!/usr/bin/env python
# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Originally part of UUV-Simulator Project. Adapted for Tartan AUV.

import math
import numpy
import rospy
import tf.transformations as trans
from PIDRegulator import PIDRegulator

import geometry_msgs.msg as geometry_msgs
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
from tauv_msgs.msg import PidVals
from tauv_msgs.srv import TunePid, TunePidResponse


class PositionControllerNode:
    def __init__(self):
        print('PositionControllerNode: initializing node')

        self.config = {}

        self.pos_des = numpy.zeros(3)
        self.quat_des = numpy.array([0, 0, 0, 1])

        self.initialized = False

        # Initialize pids with default parameters
        l_p = rospy.get_param("~pos_p")
        l_i = rospy.get_param("~pos_i")
        l_d = rospy.get_param("~pos_d")
        l_sat = rospy.get_param("~pos_sat")
        a_p = rospy.get_param("~rot_p")
        a_i = rospy.get_param("~rot_i")
        a_d = rospy.get_param("~rot_d")
        a_sat = rospy.get_param("~rot_sat")
        self.pid_rot = PIDRegulator(a_p, a_i, a_d, a_sat)
        self.pid_pos = PIDRegulator(l_p, l_i, l_d, l_sat)

        # ROS infrastructure
        self.cfg = PidVals()
        self.cfg.a_p = a_p
        self.cfg.a_i = a_i
        self.cfg.a_d = a_d
        self.cfg.a_sat = a_sat
        self.cfg.l_p = l_p
        self.cfg.l_i = l_i
        self.cfg.l_d = l_d
        self.cfg.l_sat = l_sat
        self.srv_reconfigure = rospy.Service("~tune", TunePid, self.reconfig_srv)
        self.pub_cfg = rospy.Publisher("~config", PidVals, queue_size=10)

        self.sub_cmd_pose = rospy.Subscriber('cmd_pose', numpy_msg(geometry_msgs.PoseStamped), self.cmd_pose_callback)
        self.sub_odometry = rospy.Subscriber('odom', numpy_msg(Odometry), self.odometry_callback)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', geometry_msgs.Twist, queue_size=10)

        self.ready = True
        print("Position Controller Started!")

    def cmd_pose_callback(self, msg):
        """Handle updated set pose callback."""
        # Just store the desired pose. The actual control runs on odometry callbacks
        p = msg.pose.position
        q = msg.pose.orientation
        self.pos_des = numpy.array([p.x, p.y, p.z])
        self.quat_des = numpy.array([q.x, q.y, q.z, q.w])

    def odometry_callback(self, msg):
        """Handle updated measured velocity callback."""
        if not self.ready:
            return

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        p = numpy.array([p.x, p.y, p.z])
        q = numpy.array([q.x, q.y, q.z, q.w])

        if not self.initialized:
            # If this is the first callback: Store and hold latest pose.
            self.pos_des = p
            self.quat_des = q
            self.initialized = True

        # Compute control output:
        t = msg.header.stamp.to_sec()

        # Position error
        e_pos_world = self.pos_des - p
        e_pos_body = trans.quaternion_matrix(q).transpose()[0:3, 0:3].dot(e_pos_world)

        # Error quaternion wrt body frame
        e_rot_quat = trans.quaternion_multiply(trans.quaternion_conjugate(q), self.quat_des)

        # Error angles
        e_rot = numpy.array(trans.euler_from_quaternion(e_rot_quat))

        v_linear = self.pid_pos.regulate(e_pos_body, t)
        v_angular = self.pid_rot.regulate(e_rot, t)

        # Convert and publish vel. command:
        cmd_vel = geometry_msgs.Twist()
        cmd_vel.linear = geometry_msgs.Vector3(*v_linear)
        cmd_vel.angular = geometry_msgs.Vector3(*v_angular)
        self.pub_cmd_vel.publish(cmd_vel)

        self.pub_cfg.publish(self.cfg)

    def reconfig_srv(self, config):
        """Handle updated configuration values."""
        # config has changed, reset PID controllers
        self.cfg = config.vals

        self.pid_pos = PIDRegulator(self.cfg.l_p, self.cfg.l_i, self.cfg.l_d, self.cfg.l_sat)
        self.pid_rot = PIDRegulator(self.cfg.a_p, self.cfg.a_i, self.cfg.a_d, self.cfg.a_sat)

        return TunePidResponse(True)



def main():
    print('starting PositionControl.py')
    rospy.init_node('position_control')
    node = PositionControllerNode()
    rospy.spin()
    print('exiting')
