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

import numpy
import rospy
import geometry_msgs.msg as geometry_msgs
from nav_msgs.msg import Odometry
import tf.transformations as trans
from rospy.numpy_msg import numpy_msg

# Modules included in this package
from PIDRegulator import PIDRegulator
from tauv_common.cfg import VelocityControlConfig
from tauv_msgs.msg import PidVals
from tauv_msgs.srv import TunePid, TunePidResponse


class VelocityControllerNode:
    def __init__(self):
        print('VelocityControllerNode: initializing node')

        self.config = {}

        self.v_linear_des = numpy.zeros(3)
        self.v_angular_des = numpy.zeros(3)

        # Initialize pids with default parameters
        l_p = rospy.get_param("~linear_p")
        l_i = rospy.get_param("~linear_i")
        l_d = rospy.get_param("~linear_d")
        l_sat = rospy.get_param("~linear_sat")
        a_p = rospy.get_param("~angular_p")
        a_i = rospy.get_param("~angular_i")
        a_d = rospy.get_param("~angular_d")
        a_sat = rospy.get_param("~angular_sat")

        self.pid_angular = PIDRegulator(a_p, a_i, a_d, a_sat)
        self.pid_linear = PIDRegulator(l_p, l_i, l_d, l_sat)

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

        self.sub_cmd_vel = rospy.Subscriber('cmd_vel', numpy_msg(geometry_msgs.Twist), self.cmd_vel_callback)
        self.sub_odometry = rospy.Subscriber('odom', numpy_msg(Odometry), self.odometry_callback)
        self.pub_cmd_accel = rospy.Publisher('cmd_accel', geometry_msgs.Accel, queue_size=10)

        self.ready = True
        print("Velocity Controller Started!")

    def cmd_vel_callback(self, msg):
        """Handle updated set velocity callback."""
        # Just store the desired velocity. The actual control runs on odometry callbacks
        v_l = msg.linear
        v_a = msg.angular
        self.v_linear_des = numpy.array([v_l.x, v_l.y, v_l.z])
        self.v_angular_des = numpy.array([v_a.x, v_a.y, v_a.z])

    def odometry_callback(self, msg):
        """Handle updated measured velocity callback."""
        if not self.ready:
            return

        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular
        v_linear = numpy.array([linear.x, linear.y, linear.z])
        v_angular = numpy.array([angular.x, angular.y, angular.z])

        # if self.config['odom_vel_in_world']:
        #     # This is a temp. workaround for gazebo's pos3d plugin not behaving properly:
        #     # Twist should be provided wrt child_frame, gazebo provides it wrt world frame
        #     # see http://docs.ros.org/api/nav_msgs/html/msg/Odometry.html
        #     xyzw_array = lambda o: numpy.array([o.x, o.y, o.z, o.w])
        #     q_wb = xyzw_array(msg.pose.pose.orientation)
        #     R_bw = trans.quaternion_matrix(q_wb)[0:3, 0:3].transpose()
        #
        #     v_linear = R_bw.dot(v_linear)
        #     v_angular = R_bw.dot(v_angular)

        # Compute compute control output:
        t = msg.header.stamp.to_sec()
        e_v_linear = (self.v_linear_des - v_linear)
        e_v_angular = (self.v_angular_des - v_angular)

        a_linear = self.pid_linear.regulate(e_v_linear, t)
        a_angular = self.pid_angular.regulate(e_v_angular, t)

        # Convert and publish accel. command:
        cmd_accel = geometry_msgs.Accel()
        cmd_accel.linear = geometry_msgs.Vector3(*a_linear)
        cmd_accel.angular = geometry_msgs.Vector3(*a_angular)
        self.pub_cmd_accel.publish(cmd_accel)

        self.pub_cfg.publish(self.cfg)

    def reconfig_srv(self, config):
        """Handle updated configuration values."""
        # config has changed, reset PID controllers
        self.cfg = config.vals

        self.pid_linear = PIDRegulator(self.cfg.l_p, self.cfg.l_i, self.cfg.l_d, self.cfg.l_sat)
        self.pid_angular = PIDRegulator(self.cfg.a_p, self.cfg.a_i, self.cfg.a_d, self.cfg.a_sat)

        return TunePidResponse(True)



def main():
    print('starting VelocityControl.py')
    rospy.init_node('velocity_control')
    node = VelocityControllerNode()
    rospy.spin()
    print('exiting')
