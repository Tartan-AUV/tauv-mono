import rospy

from geometry_msgs.msg import Pose, PoseStamped, Twist, Accel, Vector3, Quaternion, Point
from nav_msgs.msg import Odometry
from scipy.spatial import transform as stf

from enum import Enum
from math import ceil


def tl(vec):
    # "To List:" Convert ros data type to list.
    if isinstance(vec, Vector3):
        return [vec.x, vec.y, vec.z]
    if isinstance(vec, Point):
        return [vec.x, vec.y, vec.z]
    if isinstance(vec, Quaternion):
        return [vec.x, vec.y, vec.z, vec.w]
    return None


class ResponseType(Enum):
    time = 1  # Measured in seconds
    acc_x = 2
    acc_y = 3
    acc_z = 4
    acc_wx = 5
    acc_wy = 6
    acc_wz = 7
    vel_x = 8
    vel_y = 9
    vel_z = 10
    vel_wx = 11
    vel_wy = 12
    vel_wz = 13
    pos_x = 14
    pos_y = 15
    pos_z = 16
    pos_wx = 17
    pos_wy = 18
    pos_wz = 19
    cmd_acc_x = 20
    cmd_acc_y = 21
    cmd_acc_z = 22
    cmd_acc_wx = 23
    cmd_acc_wy = 24
    cmd_acc_wz = 25
    cmd_vel_x = 26
    cmd_vel_y = 27
    cmd_vel_z = 28
    cmd_vel_wx = 29
    cmd_vel_wy = 30
    cmd_vel_wz = 31
    cmd_pos_x = 32
    cmd_pos_y = 33
    cmd_pos_z = 34
    cmd_pos_wx = 35
    cmd_pos_wy = 36
    cmd_pos_wz = 37


class ResponseModel:
    def __init__(self):

        self.max_preservation_time = 30.0

        self.accel_alpha = 0.0

        self.dt = 0.01

        self.last_odom = Odometry()
        self.last_vel = None
        self.last_accel = None

        self.sub_odom = None
        self.sub_cmd_acc = None
        self.sub_cmd_vel = None
        self.sub_cmd_pos = None

        self.last_cmd_pos = PoseStamped()
        self.last_cmd_vel = Twist()
        self.last_cmd_acc = Accel()

        self.startup_time = rospy.Time.now().to_sec()

        self.responses = [r.value for r in ResponseType]

        self.buffers = {r: [0.0] * int(ceil((self.max_preservation_time / self.dt))) for r in ResponseType}

        for r in self.responses:
            buf = [0] * int(ceil(1.0 / self.dt * self.max_preservation_time))
            self.buffers[r] = buf

        rospy.Timer(rospy.Duration(self.dt), self.update)

    def declare_subscribers(self, topics):
        if self.sub_cmd_acc is not None:
            self.sub_cmd_acc.unregister()
        if self.sub_cmd_vel is not None:
            self.sub_cmd_vel.unregister()
        if self.sub_cmd_pos is not None:
            self.sub_cmd_pos.unregister()
        if self.sub_odom is not None:
            self.sub_odom.unregister()

        self.sub_odom = rospy.Subscriber(topics["odom"], Odometry, self.odom_callback)
        self.sub_cmd_acc = rospy.Subscriber(topics["cmd_acc"], Accel, self.cmd_acc_callback)
        self.sub_cmd_vel = rospy.Subscriber(topics["cmd_vel"], Twist, self.cmd_vel_callback)
        self.sub_cmd_pos = rospy.Subscriber(topics["cmd_pos"], PoseStamped, self.cmd_pos_callback)

    def add_pt(self, r, v):
        self.buffers[r].pop(0)
        self.buffers[r].append(v)

    def odom_callback(self, odom):
        self.last_odom = odom

    def cmd_acc_callback(self, cmd_acc):
        self.last_cmd_vel = cmd_acc

    def cmd_vel_callback(self, cmd_vel):
        self.last_cmd_vel = cmd_vel

    def cmd_pos_callback(self, cmd_pos):
        self.last_cmd_pos = cmd_pos

    def update(self, timer_event):
        t = rospy.Time.now()
        self.add_pt(ResponseType.time, t.to_sec() - self.startup_time)

        # Do Odom data:

        # position
        pose = self.last_odom.pose.pose
        try:
            zyx = stf.Rotation.from_quat(tl(pose.orientation)).as_euler("ZYX")
        except:
            zyx = [0, 0, 0]

        self.add_pt(ResponseType.pos_x, pose.position.x)
        self.add_pt(ResponseType.pos_y, pose.position.y)
        self.add_pt(ResponseType.pos_z, pose.position.z)
        self.add_pt(ResponseType.pos_wx, zyx[2])
        self.add_pt(ResponseType.pos_wy, zyx[1])
        self.add_pt(ResponseType.pos_wz, zyx[0])

        # velocity
        twist = self.last_odom.twist.twist

        self.add_pt(ResponseType.vel_x, twist.linear.x)
        self.add_pt(ResponseType.vel_y, twist.linear.y)
        self.add_pt(ResponseType.vel_z, twist.linear.z)
        self.add_pt(ResponseType.vel_wx, twist.angular.x)
        self.add_pt(ResponseType.vel_wy, twist.angular.y)
        self.add_pt(ResponseType.vel_wz, twist.angular.z)

        # acceleration
        if self.last_vel is None:
            # Prevents acceleration spike at the beginning
            self.last_vel = twist

        accel = Accel()
        accel.linear.x = (twist.linear.x - self.last_vel.linear.x) / self.dt
        accel.linear.y = (twist.linear.y - self.last_vel.linear.y) / self.dt
        accel.linear.z = (twist.linear.z - self.last_vel.linear.z) / self.dt
        accel.angular.x = (twist.angular.x - self.last_vel.angular.x) / self.dt
        accel.angular.y = (twist.angular.y - self.last_vel.angular.y) / self.dt
        accel.angular.z = (twist.angular.z - self.last_vel.angular.z) / self.dt
        self.last_vel = twist

        # low pass filter acceleration:
        if self.last_accel is None:
            # Prevents slow ramp of initial accel measurement
            self.last_accel = accel

        alph = float(self.accel_alpha)
        accel.linear.x = self.last_accel.linear.x * alph + accel.linear.x * (1.0 - alph)
        accel.linear.y = self.last_accel.linear.y * alph + accel.linear.y * (1.0 - alph)
        accel.linear.z = self.last_accel.linear.z * alph + accel.linear.z * (1.0 - alph)
        accel.angular.x = self.last_accel.angular.x * alph + accel.angular.x * (1.0 - alph)
        accel.angular.y = self.last_accel.angular.y * alph + accel.angular.y * (1.0 - alph)
        accel.angular.z = self.last_accel.angular.z * alph + accel.angular.z * (1.0 - alph)
        self.last_accel = accel

        self.add_pt(ResponseType.acc_x, accel.linear.x)
        self.add_pt(ResponseType.acc_y, accel.linear.y)
        self.add_pt(ResponseType.acc_z, accel.linear.z)
        self.add_pt(ResponseType.acc_wx, accel.angular.x)
        self.add_pt(ResponseType.acc_wy, accel.angular.y)
        self.add_pt(ResponseType.acc_wz, accel.angular.z)

        # Do cmd_pos
        pose = self.last_cmd_pos.pose
        try:
            zyx = stf.Rotation.from_quat(tl(pose.orientation)).as_euler("ZYX")
        except:
            zyx = [0, 0, 0]

        self.add_pt(ResponseType.cmd_pos_x, pose.position.x)
        self.add_pt(ResponseType.cmd_pos_y, pose.position.y)
        self.add_pt(ResponseType.cmd_pos_z, pose.position.z)
        self.add_pt(ResponseType.cmd_pos_wx, zyx[2])
        self.add_pt(ResponseType.cmd_pos_wy, zyx[1])
        self.add_pt(ResponseType.cmd_pos_wz, zyx[0])

        # Do cmd_vel
        twist = self.last_cmd_vel
        self.add_pt(ResponseType.cmd_vel_x, twist.linear.x)
        self.add_pt(ResponseType.cmd_vel_y, twist.linear.y)
        self.add_pt(ResponseType.cmd_vel_z, twist.linear.z)
        self.add_pt(ResponseType.cmd_vel_wx, twist.angular.x)
        self.add_pt(ResponseType.cmd_vel_wy, twist.angular.y)
        self.add_pt(ResponseType.cmd_vel_wz, twist.angular.z)

        # Do cmd_acc
        accel = self.last_cmd_acc
        self.add_pt(ResponseType.cmd_acc_x, accel.linear.x)
        self.add_pt(ResponseType.cmd_acc_y, accel.linear.y)
        self.add_pt(ResponseType.cmd_acc_z, accel.linear.z)
        self.add_pt(ResponseType.cmd_acc_wx, accel.angular.x)
        self.add_pt(ResponseType.cmd_acc_wy, accel.angular.y)
        self.add_pt(ResponseType.cmd_acc_wz, accel.angular.z)

    def get_data(self, response, windowWidth=None):
        if windowWidth is None:
            windowWidth = self.max_preservation_time
        nsamps = int(ceil(windowWidth / self.dt))
        buf_x = self.buffers[ResponseType.time][-nsamps:]
        buf_y = self.buffers[response][-nsamps:]
        return buf_x, buf_y
