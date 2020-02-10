import rospy

from geometry_msgs.msg import Pose, PoseStamped, Twist, Accel, Vector3, Quaternion, Point
from nav_msgs.msg import Odometry

from enum import Enum
from math import ceil


class ResponseType(Enum):
    time = 1 # Measured in seconds
    accel_x = 2
    accel_y = 3
    accel_z = 4
    accel_wx = 5
    accel_wy = 6
    accel_wz = 7
    vel_x = 8
    vel_y = 9
    vel_z = 10
    vel_wx = 11
    vel_wy = 12
    vel_wz = 13
    pos_x = 14
    pos_y = 15
    pos_z = 16
    cmd_accel_x = 17
    cmd_accel_y = 18
    cmd_accel_z = 19
    cmd_accel_wx = 20
    cmd_accel_wy = 21
    cmd_accel_wz = 22
    cmd_vel_x = 23
    cmd_vel_y = 24
    cmd_vel_z = 25
    cmd_vel_wx = 26
    cmd_vel_wy = 27
    cmd_vel_wz = 28
    cmd_pos_x = 29
    cmd_pos_y = 30
    cmd_pos_z = 31


class ResponseModel:
    def __init__(self):
        self.sub_odom = rospy.Subscriber(rospy.get_param("~odom_topic"), self.odom_callback)

        self.sub_cmd_acc = rospy.Subscriber(rospy.get_param("~cmd_accel_topic"), self.cmd_acc_callback)
        self.sub_cmd_vel = rospy.Subscriber(rospy.get_param("~cmd_vel_topic"), self.cmd_vel_callback)
        self.sub_cmd_pos = rospy.Subscriber(rospy.get_param("~cmd_pose_topic"), self.cmd_pos_callback)

        self.max_preservation_time = rospy.get_param("~max_preservation_time")
        self.dt = 0.01

        self.last_odom = Odometry()
        self.last_cmd_pos = PoseStamped()
        self.last_cmd_vel = Twist()
        self.last_cmd_acc = Accel()

        self.responses = [r.value for r in ResponseType]

        self.buffers = {}

        for r in self.responses:
            buf = [0] * int(ceil(1.0/self.dt * self.max_preservation_time))
            self.buffers[r] = buf

        rospy.Timer(rospy.Duration(1.0/self.dt), self.update)

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

    def update(self):
        # Do odom:
        self.add_pt(vel_x)
