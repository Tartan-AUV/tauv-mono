import rospy
import tf
import numpy as np
from scipy.spatial import transform as stf

from std_msgs.msg import Float64
from tauv_msgs.msg import ControllerInput
from geometry_msgs.msg import WrenchStamped, Wrench
from std_msgs.msg import Header


class CascadedController:
    def __init__(self):
        self.tfl = tf.TransformListener()
        self.tfb = tf.TransformBroadcaster()

        # parameters:
        model_name = rospy.get_param("model_name")
        self.base_frame = model_name + "/base_link"
        self.stab_frame = model_name + "/base_link_stabilized"
        self.world_frame = "world"# model_name + "/odom"
        self.vel_avg_interval = rospy.Duration(rospy.get_param("~velocity_averaging_interval"))

        self.enable_pos_control = [False]*6
        self.enable_vel_control = [False]*6

        # Local data:
        self.pos_roll_effort = 0
        self.pos_pitch_effort = 0
        self.pos_yaw_effort = 0
        self.pos_x_effort = 0
        self.pos_y_effort = 0
        self.pos_z_effort = 0
        self.vel_roll_effort = 0
        self.vel_pitch_effort = 0
        self.vel_yaw_effort = 0
        self.vel_x_effort = 0
        self.vel_y_effort = 0
        self.vel_z_effort = 0
        self.torque_roll = 0
        self.torque_pitch = 0
        self.torque_yaw = 0
        self.torque_x = 0
        self.torque_y = 0
        self.torque_z = 0

        # Position publishers:
        self.pub_pos_roll = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_roll_controller/topic_from_plant"),
                                            Float64, queue_size=10)
        self.pub_pos_pitch = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_pitch_controller/topic_from_plant"),
                                             Float64, queue_size=10)
        self.pub_pos_yaw = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_yaw_controller/topic_from_plant"),
                                           Float64, queue_size=10)
        self.pub_pos_x = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_x_controller/topic_from_plant"),
                                         Float64, queue_size=10)
        self.pub_pos_y = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_y_controller/topic_from_plant"),
                                         Float64, queue_size=10)
        self.pub_pos_z = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_z_controller/topic_from_plant"),
                                         Float64, queue_size=10)
        self.pub_targ_pos_roll = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_roll_controller/setpoint_topic"),
                                            Float64, queue_size=10)
        self.pub_targ_pos_pitch = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_pitch_controller/setpoint_topic"),
                                             Float64, queue_size=10)
        self.pub_targ_pos_yaw = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_yaw_controller/setpoint_topic"),
                                           Float64, queue_size=10)
        self.pub_targ_pos_x = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_x_controller/setpoint_topic"),
                                         Float64, queue_size=10)
        self.pub_targ_pos_y = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_y_controller/setpoint_topic"),
                                         Float64, queue_size=10)
        self.pub_targ_pos_z = rospy.Publisher("~pids/" + rospy.get_param("~pids/pos_z_controller/setpoint_topic"),
                                         Float64, queue_size=10)

        # Velocity publishers:
        self.pub_vel_roll = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_roll_controller/topic_from_plant"),
                                            Float64, queue_size=10)
        self.pub_vel_pitch = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_pitch_controller/topic_from_plant"),
                                             Float64, queue_size=10)
        self.pub_vel_yaw = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_yaw_controller/topic_from_plant"),
                                           Float64, queue_size=10)
        self.pub_vel_x = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_x_controller/topic_from_plant"),
                                         Float64, queue_size=10)
        self.pub_vel_y = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_y_controller/topic_from_plant"),
                                         Float64, queue_size=10)
        self.pub_vel_z = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_z_controller/topic_from_plant"),
                                         Float64, queue_size=10)
        self.pub_targ_vel_roll = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_roll_controller/setpoint_topic"),
                                            Float64, queue_size=10)
        self.pub_targ_vel_pitch = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_pitch_controller/setpoint_topic"),
                                             Float64, queue_size=10)
        self.pub_targ_vel_yaw = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_yaw_controller/setpoint_topic"),
                                           Float64, queue_size=10)
        self.pub_targ_vel_x = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_x_controller/setpoint_topic"),
                                         Float64, queue_size=10)
        self.pub_targ_vel_y = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_y_controller/setpoint_topic"),
                                         Float64, queue_size=10)
        self.pub_targ_vel_z = rospy.Publisher("~pids/" + rospy.get_param("~pids/vel_z_controller/setpoint_topic"),
                                         Float64, queue_size=10)

        # Position effort subscribers:
        self.sub_pos_roll_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/pos_roll_controller/topic_from_controller"),
                                                    Float64, self.callback_pos_roll_effort)
        self.sub_pos_pitch_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/pos_pitch_controller/topic_from_controller"),
                                                     Float64, self.callback_pos_pitch_effort)
        self.sub_pos_yaw_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/pos_yaw_controller/topic_from_controller"),
                                                   Float64, self.callback_pos_yaw_effort)
        self.sub_pos_x_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/pos_x_controller/topic_from_controller"),
                                                 Float64, self.callback_pos_x_effort)
        self.sub_pos_y_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/pos_y_controller/topic_from_controller"),
                                                 Float64, self.callback_pos_y_effort)
        self.sub_pos_z_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/pos_z_controller/topic_from_controller"),
                                                 Float64, self.callback_pos_z_effort)
        # Velocity effort subscribers:
        self.sub_vel_roll_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/vel_roll_controller/topic_from_controller"),
                                                    Float64, self.callback_vel_roll_effort)
        self.sub_vel_pitch_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/vel_pitch_controller/topic_from_controller"),
                                                     Float64, self.callback_vel_pitch_effort)
        self.sub_vel_yaw_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/vel_yaw_controller/topic_from_controller"),
                                                   Float64, self.callback_vel_yaw_effort)
        self.sub_vel_x_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/vel_x_controller/topic_from_controller"),
                                                 Float64, self.callback_vel_x_effort)
        self.sub_vel_y_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/vel_y_controller/topic_from_controller"),
                                                 Float64, self.callback_vel_y_effort)
        self.sub_vel_z_effort = rospy.Subscriber("~pids/" + rospy.get_param("~pids/vel_z_controller/topic_from_controller"),
                                                 Float64, self.callback_vel_z_effort)

        # Reference Subscriber:
        self.sub_ref = rospy.Subscriber("~" + rospy.get_param("~input_topic"), ControllerInput, self.set_ref)

        # Wrench Publisher:
        self.pub_wrench = rospy.Publisher("~" + rospy.get_param("~output_topic"), WrenchStamped, queue_size=10)

    def update(self, timerEvent):
        # Get position state:
        try:
            # pos is position in odom, rot is a quaternion representing orientation
            (pos, rot) = self.tfl.lookupTransform(self.world_frame, self.base_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # TODO: set exception here
            print("Failed to find transformation between frames: {}".format(e))
            return

        # Convert quaternion to roll,pitch,yaw. (ZYX euler angles)
        R_pos = stf.Rotation.from_quat(np.array(rot))
        ypr = R_pos.as_euler("ZYX")

        # Publish position state:
        self.pub_pos_x.publish(Float64(pos[0]))
        self.pub_pos_y.publish(Float64(pos[1]))
        self.pub_pos_z.publish(Float64(pos[2]))
        self.pub_pos_roll.publish(Float64(ypr[2]))
        self.pub_pos_pitch.publish(Float64(ypr[1]))
        self.pub_pos_yaw.publish(Float64(ypr[0]))

        # Get Angular velocity state in the inertial frame:
        try:
            # vel: velocity as x,y,z. ang_vel: angular velocity about x,y,z of odom respectively.
            (vel, ang_vel) = self.tfl.lookupTwistFull(self.base_frame,  # Tracking frame
                                                      self.base_frame,  # Observation frame
                                                      self.world_frame,  # Reference frame
                                                      [0, 0, 0],  # Reference point
                                                      self.base_frame,  # Reference point frame
                                                      rospy.Time(0),
                                                      self.vel_avg_interval)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # TODO: set exception here
            print("Failed to find transformation between frames: {}".format(e))
            return
        self.pub_vel_roll.publish(Float64(ang_vel[0]))
        self.pub_vel_pitch.publish(Float64(ang_vel[1]))
        self.pub_vel_yaw.publish(Float64(ang_vel[2]))

        # Get Linear velocity state in the odom frame:
        try:
            # vel: velocity as x,y,z. ang_vel: angular velocity about x,y,z of odom respectively.
            (vel, ang_vel) = self.tfl.lookupTwistFull(self.base_frame,  # Tracking frame
                                                      self.world_frame,  # Observation frame
                                                      self.world_frame,  # Reference frame
                                                      [0, 0, 0],  # Reference point
                                                      self.base_frame,  # Reference point frame
                                                      rospy.Time(0),
                                                      self.vel_avg_interval)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # TODO: set exception here
            print("Failed to find transformation between frames: {}".format(e))
            return
        self.pub_vel_x.publish(Float64(vel[0]))
        self.pub_vel_y.publish(Float64(vel[1]))
        self.pub_vel_z.publish(Float64(vel[2]))

        # Publish the stabilized frame:
        R_stab = stf.Rotation.from_euler("ZYX", [ypr[0], 0, 0])
        quat = R_stab.as_quat()
        self.tfb.sendTransform(pos, quat, rospy.Time.now(), self.stab_frame, self.world_frame)

        # Calculate and publish resultant wrench:
        f_odom = [0, 0, 0, 0, 0, 0]
        f_base = [0, 0, 0, 0, 0, 0]

        # Add up efforts:
        # Position Controllers:
        if self.enable_pos_control[0]:
            f_odom[0] += self.pos_x_effort
        if self.enable_pos_control[1]:
            f_odom[1] += self.pos_y_effort
        if self.enable_pos_control[2]:
            f_odom[2] += self.pos_z_effort
        if self.enable_pos_control[3]:
            f_base[3] += self.pos_roll_effort
        if self.enable_pos_control[4]:
            f_base[4] += self.pos_pitch_effort
        if self.enable_pos_control[5]:
            f_base[5] += self.pos_yaw_effort

        # Velocity controllers:
        if self.enable_vel_control[0]:
            f_odom[0] += self.vel_x_effort
        if self.enable_vel_control[1]:
            f_odom[1] += self.vel_y_effort
        if self.enable_vel_control[2]:
            f_odom[2] += self.vel_z_effort
        if self.enable_vel_control[3]:
            f_base[3] += self.vel_roll_effort
        if self.enable_vel_control[4]:
            f_base[4] += self.vel_pitch_effort
        if self.enable_vel_control[5]:
            f_base[5] += self.vel_yaw_effort

        # Feed-forward torque:
        f_odom[0] += self.torque_x
        f_odom[1] += self.torque_y
        f_odom[2] += self.torque_z
        f_base[3] += self.torque_roll
        f_base[4] += self.torque_pitch
        f_base[5] += self.torque_yaw

        # Convert odom wrench to base_link:
        force_odom = f_odom[0:3]
        torque_odom = f_odom[3:6]
        try:
            # pos is position in odom, rot is a quaternion representing orientation
            (pos, rot) = self.tfl.lookupTransform(self.base_frame, self.world_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            # TODO: set exception here
            print("Failed to find transformation between frames: {}".format(e))
            return
        R = stf.Rotation.from_quat(rot)
        force_base = np.dot(R.as_dcm(), np.array(force_odom))
        torque_base = np.dot(R.as_dcm(), np.array(torque_odom))

        print("from_base: {}, from_odom: {}, torque_base: {}".format(f_base, force_base, torque_base))

        ws = WrenchStamped()
        ws.header = Header()
        ws.header.stamp = rospy.Time.now()
        ws.header.frame_id = self.base_frame
        ws.wrench.force.x = f_base[0] + force_base[0]
        ws.wrench.force.y = f_base[1] + force_base[1]
        ws.wrench.force.z = f_base[2] + force_base[2]
        ws.wrench.torque.x = f_base[3] + torque_base[0]
        ws.wrench.torque.y = f_base[4] + torque_base[1]
        ws.wrench.torque.z = f_base[5] + torque_base[2]

        self.pub_wrench.publish(ws)

    def set_ref(self, msg):
        self.enable_pos_control = msg.enable_pos_control
        self.enable_vel_control = msg.enable_vel_control

        # Convert quaternion to inertial r,p,y
        ori = msg.pos_target[3:7]
        R = stf.Rotation.from_quat(ori)
        ZYX = R.as_euler("ZYX")

        # Publish position setpoints:
        self.pub_targ_pos_x.publish(Float64(msg.pos_target[0]))  # position in x  (odom frame)
        self.pub_targ_pos_y.publish(Float64(msg.pos_target[1]))  # position in y  (odom frame)
        self.pub_targ_pos_z.publish(Float64(msg.pos_target[2]))  # position in z  (odom frame)
        self.pub_targ_pos_roll.publish(Float64(ZYX[2]))  # inertial roll  (inertial frame)
        self.pub_targ_pos_pitch.publish(Float64(ZYX[1]))  # inertial pitch (inertial frame)
        self.pub_targ_pos_yaw.publish(Float64(ZYX[0]))  # inertial yaw   (inertial frame)

        # Convert angular velocity setpoints into inertial frame if necessary:
        if msg.use_inertial_ang_vel:
            w = msg.vel_target[3:6]
        else:
            w = msg.vel_target[3:6]
            try:
                # pos is position in odom, rot is a quaternion representing orientation
                (pos, rot) = self.tfl.lookupTransform(self.world_frame, self.base_frame, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                # TODO: set exception here
                # print("Failed to find transformation between frames: {}".format(e))
                return
            R = stf.Rotation.from_quat(rot)
            w = np.dot(R.as_dcm(), np.array(w))

        # Publish velocity setpoints
        self.pub_targ_vel_x.publish(Float64(msg.vel_target[0]))  # velocity in x  (odom frame)
        self.pub_targ_vel_y.publish(Float64(msg.vel_target[1]))  # velocity in y  (odom frame)
        self.pub_targ_vel_z.publish(Float64(msg.vel_target[2]))  # velocity in z  (odom frame)
        self.pub_targ_vel_roll.publish(Float64(w[0]))  # roll velocity  (inertial frame)
        self.pub_targ_vel_pitch.publish(Float64(w[1]))  # pitch velocity (inertial frame)
        self.pub_targ_vel_yaw.publish(Float64(w[2]))  # yaw velocity   (inertial frame)

        # Set feedforward torques: (odom)
        self.torque_x = msg.torque_ff[0]
        self.torque_y = msg.torque_ff[1]
        self.torque_z = msg.torque_ff[2]
        self.torque_roll = msg.torque_ff[3]
        self.torque_pitch = msg.torque_ff[4]
        self.torque_yaw = msg.torque_ff[5]

    # Effort callbacks
    def callback_pos_roll_effort(self, msg):
        self.pos_roll_effort = msg.data

    def callback_pos_pitch_effort(self, msg):
        self.pos_pitch_effort = msg.data

    def callback_pos_yaw_effort(self, msg):
        self.pos_yaw_effort = msg.data

    def callback_pos_x_effort(self, msg):
        self.pos_x_effort = msg.data

    def callback_pos_y_effort(self, msg):
        self.pos_y_effort = msg.data

    def callback_pos_z_effort(self, msg):
        self.pos_z_effort = msg.data

    def callback_vel_roll_effort(self, msg):
        self.vel_roll_effort = msg.data

    def callback_vel_pitch_effort(self, msg):
        self.vel_pitch_effort = msg.data

    def callback_vel_yaw_effort(self, msg):
        self.vel_yaw_effort = msg.data

    def callback_vel_x_effort(self, msg):
        self.vel_x_effort = msg.data

    def callback_vel_y_effort(self, msg):
        self.vel_y_effort = msg.data

    def callback_vel_z_effort(self, msg):
        self.vel_z_effort = msg.data

    def start(self):
        rospy.Timer(rospy.Duration(1.0 / 100), self.update)
        rospy.spin()

def main():
    rospy.init_node('cascaded_controller')
    cc = CascadedController()
    cc.start()
