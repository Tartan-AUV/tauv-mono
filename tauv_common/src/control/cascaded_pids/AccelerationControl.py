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

from __future__ import print_function
import numpy
import rospy

from std_msgs.msg import Bool
from geometry_msgs.msg import Accel, Vector3
from geometry_msgs.msg import Wrench
from rospy.numpy_msg import numpy_msg
from tauv_msgs.msg import InertialVals
from tauv_msgs.srv import TuneInertial, TuneInertialResponse

import tf
from scipy.spatial import transform as stf


def tl(vec3):
    # "To List:" Convert vector3 to list.
    return [vec3.x, vec3.y, vec3.z]


class AccelerationControllerNode:
    def __init__(self):
        print('AccelerationControllerNode: initializing node')

        self.ready = False
        self.mass = 1.
        self.inertial_tensor = numpy.identity(3)
        self.mass_inertial_matrix = numpy.zeros((6, 6))

        self.enable_bc = False

        self.R = stf.Rotation((0, 0, 0, 1))
        self.R_inv = stf.Rotation((0, 0, 0, 1))

        # Setup tf listener:
        self.tfl = tf.TransformListener()
        self.body = 'base_link'
        self.odom = 'odom'

        # ROS infrastructure
        self.pub_gen_force = rospy.Publisher(
          'thruster_manager/input', Wrench, queue_size=1)

        if not rospy.has_param("pid/mass"):
            raise rospy.ROSException("UUV's mass was not provided")

        if not rospy.has_param("pid/inertial"):
            raise rospy.ROSException("UUV's inertial was not provided")

        self.mass = rospy.get_param("pid/mass")
        self.buoyancy = rospy.get_param("pid/buoyancy")
        inertial = rospy.get_param("pid/inertial")

        # update mass, moments of inertia
        self.inertial_tensor = numpy.array(
          [[inertial['ixx'], inertial['ixy'], inertial['ixz']],
           [inertial['ixy'], inertial['iyy'], inertial['iyz']],
           [inertial['ixz'], inertial['iyz'], inertial['izz']]])
        self.mass_inertial_matrix = numpy.vstack((
          numpy.hstack((self.mass*numpy.identity(3), numpy.zeros((3, 3)))),
          numpy.hstack((numpy.zeros((3, 3)), self.inertial_tensor))))

        self.cfg = InertialVals()
        self.cfg.mass = self.mass
        self.cfg.buoyancy = self.buoyancy
        self.cfg.ixx = inertial['ixx']
        self.cfg.iyy = inertial['iyy']
        self.cfg.izz = inertial['izz']

        self.pub_cfg = rospy.Publisher("~config", InertialVals, queue_size=10)

        self.force_msg = None

        self.srv_reconfigure = rospy.Service("~tune", TuneInertial, self.reconfig_srv)

        self.sub_accel = rospy.Subscriber('cmd_accel', numpy_msg(Accel), self.accel_callback)
        self.sub_enablebc = rospy.Subscriber('enable_bc', Bool, self.enable_bc_callback)
        print(self.mass_inertial_matrix)
        self.ready = True

    def accel_callback(self, msg):
        if not self.ready:
            return

        try:
            (self.pos, self.orientation) = self.tfl.lookupTransform(self.odom, self.body, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print("Failed to find transformation between frames: {}".format(e))
            return
        self.R = stf.Rotation.from_quat(self.orientation)  # Transformation matrix from body to odom
        self.R_inv = self.R.inv()

        # extract 6D accelerations (linear, angular) from message
        linear = numpy.array((msg.linear.x, msg.linear.y, msg.linear.z))
        angular = numpy.array((msg.angular.x, msg.angular.y, msg.angular.z))
        accel = numpy.hstack((linear, angular)).transpose()

        if self.enable_bc:
            buoyancy_force = [0, 0, self.buoyancy]
        else:
            buoyancy_force = [0, 0, 0]
        buoyancy_force_body = self.odom2body(buoyancy_force)

        # convert acceleration to force / torque
        force_torque = self.mass_inertial_matrix.dot(accel)

        force_msg = Wrench()
        force_msg.force.x = force_torque[0] + buoyancy_force_body[0]
        force_msg.force.y = force_torque[1] + buoyancy_force_body[0]
        force_msg.force.z = force_torque[2] + buoyancy_force_body[0]

        force_msg.torque.x = force_torque[3]
        force_msg.torque.y = force_torque[4]
        force_msg.torque.z = force_torque[5]

        self.pub_cfg.publish(self.cfg)

        self.pub_gen_force.publish(force_msg)

    def body2odom(self, vec):
        if isinstance(vec, Vector3):
            vec = tl(vec)
        return self.R.apply(vec)

    def odom2body(self, vec):
        if isinstance(vec, Vector3):
            vec = tl(vec)
        return self.R_inv.apply(vec)

    def reconfig_srv(self, req):
        self.cfg = req.vals

        self.mass = req.vals.mass
        self.buoyancy = req.vals.buoyancy

        # update mass, moments of inertia
        self.inertial_tensor = numpy.array(
            [[req.vals.ixx, 0.0, 0.0],
             [0.0, req.vals.iyy, 0.0],
             [0.0, 0.0, req.vals.izz]])
        self.mass_inertial_matrix = numpy.vstack((
            numpy.hstack((self.mass*numpy.identity(3), numpy.zeros((3, 3)))),
            numpy.hstack((numpy.zeros((3, 3)), self.inertial_tensor))))

        return TuneInertialResponse(True)

    def enable_bc_callback(self, msg):
        self.enable_bc = msg.data


def main():
    print('starting AccelerationControl.py')
    rospy.init_node('acceleration_control')
    node = AccelerationControllerNode()
    rospy.spin()
    print('exiting')
