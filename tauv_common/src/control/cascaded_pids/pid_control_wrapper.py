# Wraps the cascaded pid controllers and allows input selections

import rospy

from tauv_msgs.msg import CascadedPidSelection
from tauv_msgs.srv import SetCascadedPidSelection, SetCascadedPidSelectionResponse
from geometry_msgs.msg import Pose, PoseStamped, Twist, Accel, Vector3, Quaternion, Point
from std_msgs.msg import Header

import tf
import numpy as np
from scipy.spatial import transform as stf

SOURCE_JOY = CascadedPidSelection.JOY
SOURCE_CONTROLLER = CascadedPidSelection.CONTROLLER


def parse(str):
    if str == "controller":
        return SOURCE_CONTROLLER
    elif str == "joy":
        return SOURCE_JOY
    raise ValueError("YAML Selections must be \"controller\" or \"joy\"")


def tl(vec3):
    # "To List:" Convert vector3 to list.
    return [vec3.x, vec3.y, vec3.z]


def tv(vec):
    return Vector3(vec[0], vec[1], vec[2])


def tp(vec):
    return Point(vec[0], vec[1], vec[2])


def tq(vec):
    return Quaternion(vec[0], vec[1], vec[2], vec[3])


class PidControlWrapper:
    def __init__(self):
        self.selections = CascadedPidSelection()
        self.load_default_config()

        self.R = stf.Rotation((0, 0, 0, 1))
        self.R_inv = stf.Rotation((0, 0, 0, 1))

        # State vars:
        self.joy_acc = None
        self.control_acc = None
        self.joy_vel = None
        self.control_vel = None
        self.joy_pos = None
        self.control_pos = None

        self.pos = None
        self.orientation = None

        # Setup tf listener:
        self.tfl = tf.TransformListener()
        self.body = 'base_link'
        self.odom = 'odom'

        # Declare publishers:
        self.pub_cmd_pos = rospy.Publisher("cmd_pose", PoseStamped, queue_size=10)
        self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.pub_cmd_acc = rospy.Publisher("cmd_accel", Accel, queue_size=10)

        # Declare status publisher:
        self.pub_status = rospy.Publisher("controller_configuration", CascadedPidSelection, queue_size=10)

        # Declare reconfiguration service:
        self.srv_config = rospy.Service("configure_controller", SetCascadedPidSelection, self.configure)

        # Declare subscribers:
        self.sub_joy_pos = rospy.Subscriber("joy_cmd_pos", PoseStamped, self.callback_cmd_pos,
                                            callback_args=SOURCE_JOY)
        self.sub_joy_vel = rospy.Subscriber("joy_cmd_vel", Twist, self.callback_cmd_vel,
                                            callback_args=SOURCE_JOY)
        self.sub_joy_acc = rospy.Subscriber("joy_cmd_acc", Accel, self.callback_cmd_acc,
                                            callback_args=SOURCE_JOY)
        self.sub_control_pos = rospy.Subscriber("guidance_cmd_pos", PoseStamped, self.callback_cmd_pos,
                                                callback_args=SOURCE_CONTROLLER)
        self.sub_control_vel = rospy.Subscriber("controller_cmd_vel", Twist, self.callback_cmd_vel,
                                                callback_args=SOURCE_CONTROLLER)
        self.sub_control_acc = rospy.Subscriber("controller_cmd_acc", Accel, self.callback_cmd_acc,
                                                callback_args=SOURCE_CONTROLLER)

    def body2odom(self, vec):
        if isinstance(vec, Vector3):
            vec = tl(vec)
        return self.R.apply(vec)

    def odom2body(self, vec):
        if isinstance(vec, Vector3):
            vec = tl(vec)
        return self.R_inv.apply(vec)

    def load_default_config(self):
        self.selections.enableBuoyancyComp = rospy.get_param("~enableBuoyancyComp")
        self.selections.enableVelocityFeedForward = rospy.get_param("~enableVelocityFeedForward")
        self.selections.pos_src_z = parse(rospy.get_param("~pos_src_z"))
        self.selections.pos_src_xy = parse(rospy.get_param("~pos_src_xy"))
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

    def configure(self, config):
        if config.reset:
            self.load_default_config()
            return SetCascadedPidSelectionResponse(True)

        valid_options = [SOURCE_JOY, SOURCE_CONTROLLER]
        if config.sel.pos_src_z not in valid_options or \
                config.sel.pos_src_xy not in valid_options or \
                config.sel.pos_src_heading not in valid_options or \
                config.sel.pos_src_attitude not in valid_options or \
                config.sel.vel_src_xy not in valid_options or \
                config.sel.vel_src_z not in valid_options or \
                config.sel.vel_src_heading not in valid_options or \
                config.sel.vel_src_attitude not in valid_options or \
                config.sel.acc_src_xy not in valid_options or \
                config.sel.acc_src_z not in valid_options or \
                config.sel.acc_src_heading not in valid_options or \
                config.sel.acc_src_attitude not in valid_options:
            return SetCascadedPidSelectionResponse(False)

        self.selections = config.sel
        return SetCascadedPidSelectionResponse(True)

    def callback_cmd_acc(self, acc, source):
        # Acceleration is in the body frame!
        if source == SOURCE_JOY:
            self.joy_acc = acc
        elif source == SOURCE_CONTROLLER:
            self.control_acc = acc
        self.update()

    def callback_cmd_vel(self, vel, source):
        # Velocity is in the body frame!
        if source == SOURCE_JOY:
            self.joy_vel = vel
        elif source == SOURCE_CONTROLLER:
            self.control_vel = vel
        self.update()

    def callback_cmd_pos(self, pos, source):
        # Pose is in the world frame!
        if source == SOURCE_JOY:
            self.joy_pos = pos
        elif source == SOURCE_CONTROLLER:
            self.control_pos = pos
        self.update()

    def calculate_acc(self):
        # Declare outputs: Both in the body frame
        angular = Vector3(0, 0, 0)

        # ANGULAR ACCELERATION:

        # Both angular are from controller: stay in the body frame.
        if self.selections.acc_src_attitude == SOURCE_CONTROLLER and \
                self.selections.acc_src_heading == SOURCE_CONTROLLER:
            if self.control_acc is None:
                return
            angular = self.control_acc.angular

        # Both angular are from joystick: use the body frame.
        if self.selections.acc_src_attitude == SOURCE_JOY and \
                self.selections.acc_src_heading == SOURCE_JOY:
            if self.joy_acc is None:
                return
            angular = self.joy_acc.angular

        # Attitude from joy, heading from controller.
        if self.selections.acc_src_heading == SOURCE_CONTROLLER and \
                self.selections.acc_src_attitude == SOURCE_JOY:
            # Convert both inputs to stabilized (level) frame:
            if self.joy_acc is None or self.control_acc is None:
                return
            control_stab = self.body2odom(self.control_acc.angular)
            joy_stab = self.body2odom(self.joy_acc.angular)

            # Replace heading (z axis) in the joystick input with the heading from the controller
            # Note that this assumes joystick attitude is in the body frame
            joy_stab[2] = control_stab[2]

            angular = tv(self.odom2body(joy_stab))

        # Attitude from controller, heading from joystick.
        if self.selections.acc_src_heading == SOURCE_JOY and \
                self.selections.acc_src_attitude == SOURCE_CONTROLLER:
            if self.joy_acc is None or self.control_acc is None:
                return
            # Convert control input to odom frame:
            control_stab = self.body2odom(self.control_acc.angular)

            # Replace heading (z axis accel) with joystick z axis accel.
            # Note that this assumes joystick heading accel is in the odom frame.
            control_stab[2] = self.joy_acc.angular.z
            angular = tv(self.odom2body(control_stab))

        # LINEAR ACCELERATION:

        linear = Vector3(0, 0, 0)

        # Both linear are from controller: use the body frame.
        if self.selections.acc_src_xy == SOURCE_CONTROLLER and \
                self.selections.acc_src_z == SOURCE_CONTROLLER:
            if self.control_acc is None:
                return
            linear = self.control_acc.linear

        # Both linear are from joystick: use the body frame.
        if self.selections.acc_src_xy == SOURCE_JOY and \
                self.selections.acc_src_z == SOURCE_JOY:
            if self.joy_acc is None:
                return
            linear = self.joy_acc.linear

        # xy from joystick, depth from controller
        if self.selections.acc_src_xy == SOURCE_JOY and \
                self.selections.acc_src_z == SOURCE_CONTROLLER:
            if self.joy_acc is None or self.control_acc is None:
                return
            # Convert both accel to stab frame:
            control_stab = self.body2odom(self.control_acc.linear)
            joy_stab = self.body2odom(self.joy_acc.linear)

            # Overwrite joystick depth with controller depth:
            joy_stab[2] = control_stab[2]

            linear = tv(self.odom2body(joy_stab))

        # xy from controller, depth from joystick
        if self.selections.acc_src_xy == SOURCE_CONTROLLER and \
                self.selections.acc_src_z == SOURCE_JOY:
            if self.joy_acc is None or self.control_acc is None:
                return
            # Convert both accel to stab frame:
            control_stab = self.body2odom(self.control_acc.linear)
            joy_stab = self.body2odom(self.joy_acc.linear)

            # Overwrite controller depth with joystick depth:
            control_stab[2] = joy_stab[2]

            linear = tv(self.odom2body(control_stab))

        res = Accel()
        res.angular = angular
        res.linear = linear
        return res

    def calculate_vel(self):
        # Declare outputs: Both in the body frame
        angular = Vector3(0, 0, 0)

        # ANGULAR VELOCITY:

        # Both angular are from controller: stay in the body frame.
        if self.selections.vel_src_attitude == SOURCE_CONTROLLER and \
                self.selections.vel_src_heading == SOURCE_CONTROLLER:
            if self.control_vel is None:
                return
            angular = self.control_vel.angular

        # Both angular are from joystick: use the body frame.
        if self.selections.vel_src_attitude == SOURCE_JOY and \
                self.selections.vel_src_heading == SOURCE_JOY:
            if self.joy_vel is None:
                return
            angular = self.joy_vel.angular

        # Attitude from joy, heading from controller.
        if self.selections.vel_src_heading == SOURCE_CONTROLLER and \
                self.selections.vel_src_attitude == SOURCE_JOY:
            # Convert both inputs to stabilized (level) frame:
            if self.joy_vel is None or self.control_vel is None:
                return
            control_stab = self.body2odom(self.control_vel.angular)
            joy_stab = self.body2odom(self.joy_vel.angular)

            # Replace heading (z axis) in the joystick input with the heading from the controller
            # Note that this assumes joystick attitude is in the body frame
            joy_stab[2] = control_stab[2]

            angular = tv(self.odom2body(joy_stab))

        # Attitude from controller, heading from joystick.
        if self.selections.vel_src_heading == SOURCE_JOY and \
                self.selections.vel_src_attitude == SOURCE_CONTROLLER:
            if self.joy_vel is None or self.control_vel is None:
                return
            # Convert control input to odom frame:
            control_stab = self.body2odom(self.control_vel.angular)

            # Replace heading (z axis velocities) with joystick z axis velocities.
            # Note that this assumes joystick heading velocities is in the odom frame.
            control_stab[2] = self.joy_vel.angular.z
            angular = tv(self.odom2body(control_stab))

        # LINEAR VELOCITY:

        linear = Vector3(0, 0, 0)

        # Both linear are from controller: use the body frame.
        if self.selections.vel_src_xy == SOURCE_CONTROLLER and \
                self.selections.vel_src_z == SOURCE_CONTROLLER:
            if self.control_vel is None:
                return
            linear = self.control_vel.linear

        # Both linear are from joystick: use the body frame.
        if self.selections.vel_src_xy == SOURCE_JOY and \
                self.selections.vel_src_z == SOURCE_JOY:
            if self.joy_vel is None:
                return
            linear = self.joy_vel.linear

        # xy from joystick, depth from controller
        if self.selections.vel_src_xy == SOURCE_JOY and \
                self.selections.vel_src_z == SOURCE_CONTROLLER:
            if self.joy_vel is None or self.control_vel is None:
                return
            # Convert both velocity to stab frame:
            control_stab = self.body2odom(self.control_vel.linear)
            joy_stab = self.body2odom(self.joy_vel.linear)

            # Overwrite joystick depth with controller depth:
            joy_stab[2] = control_stab[2]

            linear = tv(self.odom2body(joy_stab))

        # xy from controller, depth from joystick
        if self.selections.vel_src_xy == SOURCE_CONTROLLER and \
                self.selections.vel_src_z == SOURCE_JOY:
            if self.joy_vel is None or self.control_vel is None:
                return
            # Convert both velocities to stab frame:
            control_stab = self.body2odom(self.control_vel.linear)
            joy_stab = self.body2odom(self.joy_vel.linear)

            # Overwrite controller depth with joystick depth:
            control_stab[2] = joy_stab[2]

            linear = tv(self.odom2body(control_stab))

        res = Twist()
        res.angular = angular
        res.linear = linear
        return res

    def calculate_pos(self):
        # Orientation:
        orientation = Quaternion(0, 0, 0, 1)

        # Both orientation are from controller: Pass quaternion through.
        if self.selections.pos_src_attitude == SOURCE_CONTROLLER and \
                self.selections.pos_src_heading == SOURCE_CONTROLLER:
            if self.control_pos is None:
                return
            orientation = self.control_pos.pose.orientation

        # Both orientation are from joystick: Pass quaternion through.
        if self.selections.pos_src_attitude == SOURCE_JOY and \
                self.selections.pos_src_heading == SOURCE_JOY:
            if self.joy_pos is None:
                return
            orientation = self.joy_pos.pose.orientation

        # Attitude from controller, heading from joystick.
        if self.selections.pos_src_attitude == SOURCE_CONTROLLER and \
                self.selections.pos_src_heading == SOURCE_JOY:
            if self.joy_pos is None or self.control_pos is None:
                return
            o_j = self.joy_pos.pose.orientation
            o_c = self.control_pos.pose.orientation
            zyx_j = stf.Rotation.from_quat(o_j).as_euler("ZYX")
            zyx_c = stf.Rotation.from_quat(o_c).as_euler("ZYX")

            zyx_res = [zyx_j[0], zyx_c[1], zyx_c[2]]
            orientation = tq(stf.Rotation.from_euler("ZYX", zyx_res).as_quat())

        # Attitude from joystick, heading from controller.
        if self.selections.pos_src_attitude == SOURCE_JOY and \
                self.selections.pos_src_heading == SOURCE_CONTROLLER:
            if self.joy_pos is None or self.control_pos is None:
                return
            o_j = self.joy_pos.pose.orientation
            o_c = self.control_pos.pose.orientation
            zyx_j = stf.Rotation.from_quat(o_j).as_euler("ZYX")
            zyx_c = stf.Rotation.from_quat(o_c).as_euler("ZYX")

            zyx_res = [zyx_c[0], zyx_j[1], zyx_j[2]]
            orientation = tq(stf.Rotation.from_euler("ZYX", zyx_res).as_quat())

        position = Point(0, 0, 0)

        # XY and Z position from controller
        if self.selections.pos_src_z == SOURCE_CONTROLLER and \
                self.selections.pos_src_xy == SOURCE_CONTROLLER:
            if self.control_pos is None:
                return
            position = self.control_pos.pose.position

        # XY and Z position from joystick
        if self.selections.pos_src_z == SOURCE_JOY and \
                self.selections.pos_src_xy == SOURCE_JOY:
            if self.joy_pos is None:
                print("fuck")
                return
            position = self.joy_pos.pose.position

        # XY from controller, Z from joystick
        if self.selections.pos_src_z == SOURCE_JOY and \
                self.selections.pos_src_xy == SOURCE_CONTROLLER:
            if self.joy_pos is None or self.control_pos is None:
                return
            p_j = tv(self.joy_pos.pose.position)
            p_c = tv(self.control_pos.pose.position)
            position = tp([p_c[0], p_c[1], p_j[2]])

        # XY from joy, Z from controller
        if self.selections.pos_src_z == SOURCE_CONTROLLER and \
                self.selections.pos_src_xy == SOURCE_JOY:
            if self.joy_pos is None or self.control_pos is None:
                return
            p_j = tv(self.joy_pos.pose.position)
            p_c = tv(self.control_pos.pose.position)
            position = tp([p_j[0], p_j[1], p_c[2]])

        pose = Pose(position, orientation)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "odom"
        return PoseStamped(header, pose)

    def update(self):

        try:
            (self.pos, self.orientation) = self.tfl.lookupTransform(self.odom, self.body, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # print("Failed to find transformation between frames: {}".format(e))
            return
        self.R = stf.Rotation.from_quat(self.orientation)  # Transformation matrix from body to odom
        self.R_inv = self.R.inv()

        # Calculate and publish commands:
        cmd_pos = self.calculate_pos()
        if cmd_pos is not None:
            self.pub_cmd_pos.publish(cmd_pos)

        cmd_vel = self.calculate_vel()
        if cmd_vel is not None:
            self.pub_cmd_vel.publish(cmd_vel)

        cmd_acc = self.calculate_acc()
        if cmd_acc is not None:
            self.pub_cmd_acc.publish(cmd_acc)

    def post_status(self, timer_event):
        self.pub_status.publish(self.selections)

    def start(self):
        rospy.Timer(rospy.Duration(0.5), self.post_status)
        rospy.spin()


def main():
    rospy.init_node('pid_control_wrapper')
    pcw = PidControlWrapper()
    pcw.start()
