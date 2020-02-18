# Wraps the cascaded pid controllers and allows input selections

import rospy

from tauv_msgs.msg import CascadedPidSelection
from tauv_msgs.srv import SetCascadedPidSelection, SetCascadedPidSelectionResponse
from geometry_msgs.msg import Pose, PoseStamped, Twist, Accel, Vector3, Quaternion, Point
from std_msgs.msg import Header, Bool

import tf
from scipy.spatial import transform as stf

SOURCE_PLANNER = CascadedPidSelection.PLANNER
SOURCE_CONTROLLER = CascadedPidSelection.CONTROLLER


def parse(str):
    if str == "controller":
        return SOURCE_CONTROLLER
    elif str == "planner":
        return SOURCE_PLANNER
    raise ValueError("YAML Selections must be \"controller\" or \"ext\"")


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

        self.available_planners = rospy.get_param("available_planners")
        print(self.available_planners)

        self.R = stf.Rotation((0, 0, 0, 1))
        self.R_inv = stf.Rotation((0, 0, 0, 1))

        # State vars:
        self.planner_acc = None
        self.control_acc = None
        self.planner_vel = None
        self.control_vel = None
        self.planner_pos = None

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
        self.pub_enable_bc = rospy.Publisher("enable_bc", Bool, queue_size=10)

        # Declare status publisher:
        self.pub_status = rospy.Publisher("controller_configuration", CascadedPidSelection, queue_size=10)

        # Declare reconfiguration service:
        self.srv_config = rospy.Service("configure_controller", SetCascadedPidSelection, self.configure)

        # Determine planner topics from planner list:
        planner_prefix = self.available_planners[self.selections.planner]["topic_prefix"]

        # Declare subscribers:
        self.sub_planner_pos = rospy.Subscriber("planners/" + planner_prefix + "/cmd_pos", PoseStamped,
                                                self.callback_cmd_pos, callback_args=SOURCE_PLANNER)
        self.sub_planner_vel = rospy.Subscriber("planners/" + planner_prefix + "/cmd_vel", Twist,
                                                self.callback_cmd_vel, callback_args=SOURCE_PLANNER)
        self.sub_planner_acc = rospy.Subscriber("planners/" + planner_prefix + "/cmd_acc", Accel,
                                                self.callback_cmd_acc, callback_args=SOURCE_PLANNER)
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
        self.selections.planner = rospy.get_param("~planner")
        self.selections.enableBuoyancyComp = rospy.get_param("~enableBuoyancyComp")
        self.selections.enableVelocityFeedForward = rospy.get_param("~enableVelocityFeedForward")
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

        valid_options = [SOURCE_PLANNER, SOURCE_CONTROLLER]
        if config.sel.vel_src_xy not in valid_options or \
                config.sel.vel_src_z not in valid_options or \
                config.sel.vel_src_heading not in valid_options or \
                config.sel.vel_src_attitude not in valid_options or \
                config.sel.acc_src_xy not in valid_options or \
                config.sel.acc_src_z not in valid_options or \
                config.sel.acc_src_heading not in valid_options or \
                config.sel.acc_src_attitude not in valid_options:
            return SetCascadedPidSelectionResponse(False)

        planner = config.sel.planner
        if planner not in self.available_planners.keys():
            return SetCascadedPidSelectionResponse(False)

        # Update selections
        self.selections = config.sel

        # Flush cached vals
        self.planner_acc = None
        self.control_acc = None
        self.planner_vel = None
        self.control_vel = None
        self.planner_pos = None

        # Update subscribers
        self.sub_planner_pos.unregister()
        self.sub_planner_vel.unregister()
        self.sub_planner_acc.unregister()

        planner_prefix = self.available_planners[planner]["topic_prefix"]

        self.sub_planner_pos = rospy.Subscriber("planners/" + planner_prefix + "/cmd_pos", PoseStamped,
                                                self.callback_cmd_pos, callback_args=SOURCE_PLANNER)
        self.sub_planner_vel = rospy.Subscriber("planners/" + planner_prefix + "/cmd_vel", Twist,
                                                self.callback_cmd_vel, callback_args=SOURCE_PLANNER)
        self.sub_planner_acc = rospy.Subscriber("planners/" + planner_prefix + "/cmd_acc", Accel,
                                                self.callback_cmd_acc, callback_args=SOURCE_PLANNER)

        return SetCascadedPidSelectionResponse(True)

    def callback_cmd_acc(self, acc, source):
        # Acceleration is in the body frame!
        if source == SOURCE_PLANNER:
            self.planner_acc = acc
        elif source == SOURCE_CONTROLLER:
            self.control_acc = acc
        self.update()

    def callback_cmd_vel(self, vel, source):
        # Velocity is in the body frame!
        if source == SOURCE_PLANNER:
            self.planner_vel = vel
        elif source == SOURCE_CONTROLLER:
            self.control_vel = vel
        self.update()

    def callback_cmd_pos(self, pos, source):
        # Pose is in the world frame!
        if source == SOURCE_PLANNER:
            self.planner_pos = pos
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

        # Both angular are from planner: use the body frame.
        if self.selections.acc_src_attitude == SOURCE_PLANNER and \
                self.selections.acc_src_heading == SOURCE_PLANNER:
            if self.planner_acc is None:
                return
            angular = self.planner_acc.angular

        # Attitude from planner, heading from controller.
        if self.selections.acc_src_heading == SOURCE_CONTROLLER and \
                self.selections.acc_src_attitude == SOURCE_PLANNER:
            # Convert both inputs to stabilized (level) frame:
            if self.planner_acc is None or self.control_acc is None:
                return
            control_stab = self.body2odom(self.control_acc.angular)
            planner_stab = self.body2odom(self.planner_acc.angular)

            # Replace heading (z axis) in the planner input with the heading from the controller
            # Note that this assumes planner attitude is in the body frame
            planner_stab[2] = control_stab[2]

            angular = tv(self.odom2body(planner_stab))

        # Attitude from controller, heading from planner.
        if self.selections.acc_src_heading == SOURCE_PLANNER and \
                self.selections.acc_src_attitude == SOURCE_CONTROLLER:
            if self.planner_acc is None or self.control_acc is None:
                return
            # Convert control input to odom frame:
            control_stab = self.body2odom(self.control_acc.angular)

            # Replace heading (z axis accel) with planner z axis accel.
            # Note that this assumes planner heading accel is in the odom frame.
            control_stab[2] = self.planner_acc.angular.z
            angular = tv(self.odom2body(control_stab))

        # LINEAR ACCELERATION:

        linear = Vector3(0, 0, 0)

        # Both linear are from controller: use the body frame.
        if self.selections.acc_src_xy == SOURCE_CONTROLLER and \
                self.selections.acc_src_z == SOURCE_CONTROLLER:
            if self.control_acc is None:
                return
            linear = self.control_acc.linear

        # Both linear are from planner: use the body frame.
        if self.selections.acc_src_xy == SOURCE_PLANNER and \
                self.selections.acc_src_z == SOURCE_PLANNER:
            if self.planner_acc is None:
                return
            linear = self.planner_acc.linear

        # xy from planner, depth from controller
        if self.selections.acc_src_xy == SOURCE_PLANNER and \
                self.selections.acc_src_z == SOURCE_CONTROLLER:
            if self.planner_acc is None or self.control_acc is None:
                return
            # Convert both accel to stab frame:
            control_stab = self.body2odom(self.control_acc.linear)
            planner_stab = self.body2odom(self.planner_acc.linear)

            # Overwrite planner depth with controller depth:
            planner_stab[2] = control_stab[2]

            linear = tv(self.odom2body(planner_stab))

        # xy from controller, depth from planner
        if self.selections.acc_src_xy == SOURCE_CONTROLLER and \
                self.selections.acc_src_z == SOURCE_PLANNER:
            if self.planner_acc is None or self.control_acc is None:
                return
            # Convert both accel to stab frame:
            control_stab = self.body2odom(self.control_acc.linear)
            planner_stab = self.body2odom(self.planner_acc.linear)

            # Overwrite controller depth with planner depth:
            control_stab[2] = planner_stab[2]

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

        # Both angular are from planner: use the body frame.
        if self.selections.vel_src_attitude == SOURCE_PLANNER and \
                self.selections.vel_src_heading == SOURCE_PLANNER:
            if self.planner_vel is None:
                return
            angular = self.planner_vel.angular

        # Attitude from planner, heading from controller.
        if self.selections.vel_src_heading == SOURCE_CONTROLLER and \
                self.selections.vel_src_attitude == SOURCE_PLANNER:
            # Convert both inputs to stabilized (level) frame:
            if self.planner_vel is None or self.control_vel is None:
                return
            control_stab = self.body2odom(self.control_vel.angular)
            planner_stab = self.body2odom(self.planner_vel.angular)

            # Replace heading (z axis) in the planner input with the heading from the controller
            # Note that this assumes planner attitude is in the body frame
            planner_stab[2] = control_stab[2]

            angular = tv(self.odom2body(planner_stab))

        # Attitude from controller, heading from planner.
        if self.selections.vel_src_heading == SOURCE_PLANNER and \
                self.selections.vel_src_attitude == SOURCE_CONTROLLER:
            if self.planner_vel is None or self.control_vel is None:
                return
            # Convert control input to odom frame:
            control_stab = self.body2odom(self.control_vel.angular)

            # Replace heading (z axis velocities) with planner z axis velocities.
            # Note that this assumes planner heading velocities is in the odom frame.
            control_stab[2] = self.planner_vel.angular.z
            angular = tv(self.odom2body(control_stab))

        # LINEAR VELOCITY:

        linear = Vector3(0, 0, 0)

        # Both linear are from controller: use the body frame.
        if self.selections.vel_src_xy == SOURCE_CONTROLLER and \
                self.selections.vel_src_z == SOURCE_CONTROLLER:
            if self.control_vel is None:
                return
            linear = self.control_vel.linear

        # Both linear are from planner: use the body frame.
        if self.selections.vel_src_xy == SOURCE_PLANNER and \
                self.selections.vel_src_z == SOURCE_PLANNER:
            if self.planner_vel is None:
                return
            linear = self.planner_vel.linear

        # xy from planner, depth from controller
        if self.selections.vel_src_xy == SOURCE_PLANNER and \
                self.selections.vel_src_z == SOURCE_CONTROLLER:
            if self.planner_vel is None or self.control_vel is None:
                return
            # Convert both velocity to stab frame:
            control_stab = self.body2odom(self.control_vel.linear)
            planner_stab = self.body2odom(self.planner_vel.linear)

            # Overwrite planner depth with controller depth:
            planner_stab[2] = control_stab[2]

            linear = tv(self.odom2body(planner_stab))

        # xy from controller, depth from planner
        if self.selections.vel_src_xy == SOURCE_CONTROLLER and \
                self.selections.vel_src_z == SOURCE_PLANNER:
            if self.planner_vel is None or self.control_vel is None:
                return
            # Convert both velocities to stab frame:
            control_stab = self.body2odom(self.control_vel.linear)
            planner_stab = self.body2odom(self.planner_vel.linear)

            # Overwrite controller depth with planner depth:
            control_stab[2] = planner_stab[2]

            linear = tv(self.odom2body(control_stab))

        res = Twist()
        res.angular = angular
        res.linear = linear
        return res

    def calculate_pos(self):
        return self.planner_pos

    def update(self):
        # Update state for transformations
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
        else:
            # Publish zero-twist to stop robot
            self.pub_cmd_vel.publish(Twist())

        cmd_acc = self.calculate_acc()
        if cmd_acc is not None:
            self.pub_cmd_acc.publish(cmd_acc)

        self.pub_enable_bc.publish(Bool(self.selections.enableBuoyancyComp))

    def post_status(self, timer_event):
        self.selections.available_planners = self.available_planners.keys()
        self.pub_status.publish(self.selections)

    def start(self):
        rospy.Timer(rospy.Duration(0.5), self.post_status)
        rospy.spin()


def main():
    rospy.init_node('pid_control_wrapper')
    pcw = PidControlWrapper()
    pcw.start()
