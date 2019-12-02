import rospy
import numpy as np
from scipy.spatial import transform as stf
import tf

from tauv_msgs.msg import ControllerInput
from tauv_msgs.srv import JoyMode, JoyModeResponse
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
        output_topic = rospy.get_param("~output_topic")

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
                         "rates_pos_offset_odom",
                         "rates_pos_offset_stab",
                         "rates_vel_odom",
                         "rates_vel_stab",
                         "torque_ff_odom",
                         "torque_ff_stab"]
        self.xy_mode = "none"

        self.joyVals = None

        # Function specific vals:
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        self.odom_to_base = stf.Rotation([0, 0, 0, 1.])
        self.base_to_odom = stf.Rotation([0, 0, 0, 1.])
        self.odom_to_stab = stf.Rotation([0, 0, 0, 1.])
        self.stab_to_odom = stf.Rotation([0, 0, 0, 1.])
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.dt = 1.0 / 100

        # Publishers/Subscribers
        self.sub_joy = rospy.Subscriber("/gui/joy", Joy, self.joyCallback)
        self.pub_control = rospy.Publisher(output_topic, ControllerInput, queue_size=10)

        # Services
        self.srv_joymode = rospy.Service('set_mode', JoyMode, self.changeMode)

    def update(self, timerEvent):
        self.update_transforms()

        cmd = ControllerInput()
        cmd.header = Header()
        cmd.header.stamp = rospy.Time.now()
        cmd.enable_pos_control = [False] * 6
        cmd.enable_vel_control = [False] * 6
        cmd.pos_target = [0.] * 7
        cmd.pos_target[6] = 1.
        cmd.vel_target = [0.] * 6
        cmd.torque_ff = [0.] * 6

        if self.joyVals is None:
            # No joystick message yet :/
            return

        self.add_attitude(cmd)
        self.add_depth(cmd)
        self.add_xy(cmd)

        self.pub_control.publish(cmd)

    def add_attitude(self, cmd):
        # Yaw
        if self.yaw_mode == "rates_pos_offset":
            cmd.enable_pos_control[5] = True
            cmd.enable_vel_control[5] = True
            sensitivity = 1.0
            curr_yaw = self.odom_to_stab.as_euler("ZYX")[0]
            if self.joyVals.buttons[joybtns["lb"]] == 0:
                yaw_rate = self.joyVals.axes[joyaxes["rx"]] * sensitivity * np.pi
                new_yaw = curr_yaw + yaw_rate * self.dt
            else:
                yaw_rate = 0
                new_yaw = 0
            cmd.vel_target[5] = yaw_rate
            self.yaw = new_yaw

        elif self.yaw_mode == "rates_vel":
            cmd.enable_vel_control[5] = True
            sensitivity = 1.0
            if self.joyVals.buttons[joybtns["lb"]] == 0:
                vel = self.joyVals.axes[joyaxes["rx"]] * sensitivity * np.pi
            else:
                vel = 0
            cmd.vel_target[5] = vel

        elif self.yaw_mode == "absolute":
            cmd.enable_pos_control[5] = True
            if self.joyVals.buttons[joybtns["lb"]] == 0:
                self.yaw = self.joyVals.axes[joyaxes["rx"]] * np.pi

        elif self.yaw_mode == "torque_ff":
            sensitivity = 10.0
            if self.joyVals.buttons[joybtns["lb"]] == 0:
                cmd.torque_ff[5] = self.joyVals.axes[joyaxes["rx"]] * np.pi * self.dt * sensitivity

        # If autolevel mode uses position commands, then we don't need to do this since it will be ic
        if self.yaw_mode in ["rates_pos_offset", "absolute"] and \
                self.autolevel_mode not in ["rates_pos_offset", "absolute"]:
            cmd.pos_target[3:7] = stf.Rotation.from_euler("ZYX", [self.yaw, 0, 0]).asra_quat()

        # Autolevel
        if self.autolevel_mode == "absolute":
            sensitivity = 0.5
            cmd.enable_pos_control[3] = True
            cmd.enable_pos_control[4] = True
            if self.joyVals.buttons[joybtns["lb"]] == 1:
                self.roll = -self.joyVals.axes[joyaxes["rx"]] * np.pi * sensitivity
                self.pitch = self.joyVals.axes[joyaxes["ry"]] * np.pi * sensitivity

            R = stf.Rotation.from_euler("ZYX", [self.yaw, self.pitch, self.roll])
            cmd.pos_target[3:7] = R.as_quat()

        elif self.autolevel_mode == "rates_vel":
            sensitivity = 0.25
            cmd.enable_vel_control[3] = True
            cmd.enable_vel_control[4] = True

            if self.joyVals.buttons[joybtns["lb"]] == 1:
                roll_vel = -self.joyVals.axes[joyaxes["rx"]] * np.pi * sensitivity
                pitch_vel = self.joyVals.axes[joyaxes["ry"]] * np.pi * sensitivity
            else:
                roll_vel = 0
                pitch_vel = 0

            w = np.array([roll_vel, pitch_vel, 0])
            w_odom = np.dot(self.odom_to_base.as_dcm(), w)
            cmd.vel_target[3] = w_odom[0]
            cmd.vel_target[4] = w_odom[1]

        elif self.autolevel_mode == "rates_pos_offset":
            sensitivity = 0.25

            cmd.enable_pos_control[3] = True
            cmd.enable_pos_control[4] = True
            cmd.enable_vel_control[3] = True
            cmd.enable_vel_control[4] = True

            if self.joyVals.buttons[joybtns["lb"]] == 1:
                roll_vel = -self.joyVals.axes[joyaxes["rx"]] * np.pi * sensitivity
                pitch_vel = self.joyVals.axes[joyaxes["ry"]] * np.pi * sensitivity
                self.roll += roll_vel * self.dt
                self.pitch += pitch_vel * self.dt
            else:
                roll_vel = 0
                pitch_vel = 0

            w = np.array([roll_vel, pitch_vel, 0])
            w_odom = np.dot(self.odom_to_base.as_dcm(), w)
            cmd.vel_target[3] = w_odom[0]
            cmd.vel_target[4] = w_odom[1]

    def add_depth(self, cmd):
        pass

    def add_xy(self, cmd):

        if self.xy_mode == "rates_pos_offset_stab":
            sensitivity = 1.0
            cmd.enable_pos_control[0] = True
            cmd.enable_pos_control[1] = True
            cmd.enable_vel_control[0] = True
            cmd.enable_vel_control[1] = True

            vel = np.array([0., 0., 0.])
            vel[0] = self.joyVals.axes[joyaxes["ly"]] * sensitivity
            vel[1] = self.joyVals.axes[joyaxes["lx"]] * sensitivity
            vel_odom = self.stab_to_odom.as_dcm().dot(vel)
            self.x += vel_odom[0] * self.dt
            self.y += vel_odom[1] * self.dt

            cmd.pos_target[0] = self.x
            cmd.pos_target[1] = self.y
            cmd.vel_target[0] = vel_odom[0]
            cmd.vel_target[1] = vel_odom[1]

        elif self.xy_mode == "rates_pos_offset_odom":
            sensitivity = 1.0
            cmd.enable_pos_control[0] = True
            cmd.enable_pos_control[1] = True
            cmd.enable_vel_control[0] = True
            cmd.enable_vel_control[1] = True

            vel = np.array([0., 0., 0.])
            vel[0] = self.joyVals.axes[joyaxes["ly"]] * sensitivity
            vel[1] = self.joyVals.axes[joyaxes["lx"]] * sensitivity
            self.x += vel[0] * self.dt
            self.y += vel[1] * self.dt

            cmd.pos_target[0] = self.x
            cmd.pos_target[1] = self.y
            cmd.vel_target[0] = vel[0]
            cmd.vel_target[1] = vel[1]

        elif self.xy_mode == "rates_vel_stab":
            sensitivity = 1.0
            cmd.enable_vel_control[0] = True
            cmd.enable_vel_control[1] = True

            vel = np.array([0., 0., 0.])
            vel[0] = self.joyVals.axes[joyaxes["ly"]] * sensitivity
            vel[1] = self.joyVals.axes[joyaxes["lx"]] * sensitivity
            vel_odom = self.stab_to_odom.as_dcm().dot(vel)

            cmd.vel_target[0] = vel_odom[0]
            cmd.vel_target[1] = vel_odom[1]

        elif self.xy_mode == "rates_vel_odom":
            sensitivity = 1.0
            cmd.enable_vel_control[0] = True
            cmd.enable_vel_control[1] = True

            vel = np.array([0., 0., 0.])
            vel[0] = self.joyVals.axes[joyaxes["ly"]] * sensitivity
            vel[1] = self.joyVals.axes[joyaxes["lx"]] * sensitivity

            cmd.vel_target[0] = vel[0]
            cmd.vel_target[1] = vel[1]

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
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

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
        return True

    def joyCallback(self, msg):
        self.joyVals = msg

    def start(self):
        rospy.Timer(rospy.Duration(self.dt), self.update)
        rospy.spin()


def main():
    rospy.init_node('cascaded_controller')
    cc = JoystickControlMode()
    cc.start()
