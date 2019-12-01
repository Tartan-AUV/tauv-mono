import rospy
import numpy as np
from scipy.spatial import transform as stf
import tf

from tauv_msgs.msg import ControllerInput
from tauv_msgs.msg import JoyMode
from std_msgs.msg import Header
from sensor_msgs.msg import Joy

joyaxes = {"lx": 0,
           "ly": 1,
           "rx": 3,
           "ry": 4,
           "z": 2,
           "dx": 6,
           "dy": 7}

joybtns = {"a": 0,
           "b": 1,
           "x": 2,
           "y": 3,
           "lb": 4,
           "rb": 5,
           "back": 6,
           "start": 7,
           "lpress": 8,
           "rpress": 9}


class JoystickControlMode:
    def __init__(self):
        self.tfl = tf.TransformListener()

        # Parameters:
        self.odom = rospy.get_param("~odom_frame")
        self.base = rospy.get_param("~base_link_frame")
        self.stab = rospy.get_param("~stab_frame")

        # Modes

        self.yaw_modes = ["none",
                          "absolute",
                          "rates_pos_offset",
                          "torque_ff",
                          "rates_vel"]
        self.yaw_mode = "none"
        self.autolevel_modes = ["none",
                                "absolute",
                                "rates_vel",
                                "rates_pos_offset"]
        self.autolevel_mode = "none"

        self.depth_modes = ["none",
                            "rates_vel",
                            "rates_pos_offset"]
        self.depth_mode = "none"

        self.xy_modes = ["none",
                         "rates_pos_offset_baselink",
                         "rates_pos_offset_stab",
                         "rates_vel_baselink",
                         "rates_vel_stab",
                         "torque_ff_baselink",
                         "torque_ff_stab"]
        self.xy_mode = "none"

        self.joyVals = Joy()

        # Function specific vals:
        self.roll = 0
        self.pitch = 0
        self.yaw_pos = 0

        self.odom_to_base = stf.Rotation([0, 0, 0, 1])
        self.odom_to_stab = stf.Rotation([0, 0, 0, 1])
        self.stab_to_odom = stf.Rotation([0, 0, 0, 1])

        self.dt = 1.0 / 100

    def update(self):
        self.update_transforms()

        cmd = ControllerInput()
        cmd.header = Header()
        cmd.header.stamp = rospy.Time.now()

        self.add_attitude(cmd)
        self.add_depth(cmd)
        self.add_xy(cmd)

    def add_attitude(self, cmd):
        # Yaw
        self.yaw_pos = 0
        if self.yaw_mode == "rates_pos_offset":
            cmd.enable_pos_control[5] = True
            sensitivity = 1
            curr_yaw = self.odom_to_stab.as_euler("ZYX")[0]
            if self.joyVals.buttons[joybtns["lb"]] == 0:
                new_yaw = curr_yaw + self.joyVals.axes[joyaxes["rx"]] * self.dt * sensitivity * np.pi
            else:
                new_yaw = 0
            self.yaw_pos = new_yaw

        elif self.yaw_mode == "rates_vel":
            cmd.enable_vel_control[5] = True
            sensitivity = 1
            if self.joyVals.buttons[joybtns["lb"]] == 0:
                vel = self.joyVals.axes[joyaxes["rx"]] * sensitivity * np.pi
            else:
                vel = 0
            cmd.vel_target[5] = vel

        elif self.yaw_mode == "absolute":
            cmd.enable_pos_control[5] = True
            if self.joyVals.buttons[joybtns["lb"]] == 0:
                self.yaw_pos = self.joyVals.axes[joyaxes["rx"]] * np.pi

        elif self.yaw_mode == "torque_ff":
            sensitivity = 10
            if self.joyVals.buttons[joybtns["lb"]] == 0:
                cmd.torque_ff[5] = self.joyVals.axes[joyaxes["rx"]] * np.pi * self.dt * sensitivity

        # If autolevel mode uses position commands, then we don't need to do this since it will be ic
        if self.yaw_mode in ["rates_pos_offset", "absolute"] and \
                self.autolevel_mode not in ["rates_pos_offset", "absolute"]:
            cmd.pos_target[3:7] = stf.Rotation.from_euler("ZYX", [0, 0, self.roll]).as_quat()

        # Autolevel
        if self.autolevel_mode == "absolute":
            cmd.enable_pos_control[3] = True
            cmd.enable_pos_control[4] = True

            if self.joyVals.buttons[joybtns["lb"]] == 1:
                self.roll = self.joyVals.axes[joyaxes["rx"]] * np.pi
                self.pitch = self.joyVals.axes[joyaxes["ry"]] * np.pi

            R = stf.Rotation.from_euler("ZYX", [self.yaw_pos, self.pitch, self.roll]).as_dcm()
            R_comb = np.dot(self.stab_to_odom.as_dcm(), R)
            quat = stf.Rotation.from_dcm(R_comb).as_quat()
            cmd.pos_target[3:7] = list(quat)

        elif self.autolevel_mode == "rates_vel":
            sensitivity = 0.25

            cmd.enable_vel_control[3] = True
            cmd.enable_vel_control[4] = True

            if self.joyVals.buttons[joybtns["lb"]] == 1:
                self.roll = self.joyVals.axes[joyaxes["rx"]] * np.pi * sensitivity
                self.pitch = self.joyVals.axes[joyaxes["ry"]] * np.pi * sensitivity

            w_base = np.array([self.roll, self.pitch, 0])
            w_odom = np.dot(self.base_to_odom, w_base)
            cmd.vel_target[3] = w_odom[0]
            cmd.vel_target[4] = w_odom[1]

    def add_depth(self, cmd):
        if self.depth_mode == "None":
            return

    def add_xy(self, cmd):
        pass

    def update_transforms(self):
        try:
            # rot is a quaternion representing orientation
            (pos, rot) = self.tfl.lookupTransform(self.odom, self.base, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # TODO: set exception here
            print("Failed to find transformation between frames: {}".format(e))
            return
        self.odom_to_base = stf.Rotation.from_quat(np.array(rot))

        try:
            # rot is a quaternion representing orientation
            (pos, rot) = self.tfl.lookupTransform(self.base, self.odom, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # TODO: set exception here
            print("Failed to find transformation between frames: {}".format(e))
            return
        self.base_to_odom = stf.Rotation.from_quat(np.array(rot))

        try:
            # rot is a quaternion representing orientation
            (pos, rot) = self.tfl.lookupTransform(self.odom, self.stab, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # TODO: set exception here
            print("Failed to find transformation between frames: {}".format(e))
            return
        self.odom_to_stab = stf.Rotation.from_quat(np.array(rot))

        try:
            # rot is a quaternion representing orientation
            (pos, rot) = self.tfl.lookupTransform(self.stab, self.odom, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # TODO: set exception here
            print("Failed to find transformation between frames: {}".format(e))
            return
        self.stab_to_odom = stf.Rotation.from_quat(np.array(rot))

    def changeMode(self, msg):
        self.yaw_mode = msg.yaw_mode
        self.depth_mode = msg.depth_mode
        self.xy_mode = msg.xy_mode
        self.autolevel_mode = msg.autolevel_mode

    def joyCallback(self, msg):
        self.joyVals = msg
