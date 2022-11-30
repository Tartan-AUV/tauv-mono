from motion.trajectories.trajectories import TrajectoryStatus
from tauv_mission_manager.mission_utils import Mission, Task, TaskParams
from tauv_mission_manager.tasks import Dive, Square, VizServo
from vision.detectors.finder import Finder
import rospy
from tauv_msgs.msg import BucketDetection, Servos
from math import sin, cos

M = 1
YD = 0.9144
FT = 0.3048
IN = FT / 12
PI = 3.14159

# POOL_LEN = 25 * YD - 1 * M
LANE_WIDTH_Y = 7 * FT
LANE_WIDTH_X = 9 * FT
POOL_LEN = 7*LANE_WIDTH_X + 6*2*FT

START_X = -6 * FT + 17 * IN + 28 * IN
START_Y = 9 * LANE_WIDTH_Y

def to_x(x_lane_lines):
    return x_lane_lines * LANE_WIDTH_X - START_X

def to_y(y_lane_lines):
    return y_lane_lines * LANE_WIDTH_Y - START_Y

# OCT_POS_X = POOL_LEN - (.5 + .67) * LANE_WIDTH_X
# OCT_POS_Y = 2.75 * LANE_WIDTH_Y

# GATE_POS_X = 1 * LANE_WIDTH_X
# GATE_POS_Y = 2.7 * LANE_WIDTH_Y

# OCT_POS_X = POOL_LEN - (.5 + .67) * LANE_WIDTH_X - 2 * M
OCT_POS_X = to_x(6.1)
OCT_POS_Y = to_y(7 + 1.4)

GATE_POS_X = to_x(1.35)
GATE_POS_Y = to_y(8.55)
GATE_ANGLE = -PI/4

BUOY_X = to_x(1.5)
BUOY_Y = to_y(4.5)
BUOY_SEARCH_X1 = to_x(4.0)
BUOY_SEARCH_Y1 = to_y(2.0)
BUOY_SEARCH_X2 = to_x(1.0)
BUOY_SEARCH_Y2 = to_y(2.0)
BUOY_SEARCH_DEPTH = 1.5
BUOY_LOOK_ANGLE = PI/2
BUOYTAG = "badge"

MARKER_X = to_x(4.5)
MARKER_Y = to_y(9.7 + 0.8)


class PoolMission(Mission):
    x = 3
    def __init__(self, params: TaskParams) -> None:
        self.p: TaskParams = params
        self.mu = params.mu
        self.p.status("I'm being born :O")
        self.dive = Dive(params)
        self.finder = Finder()

        self.servo_pub = rospy.Publisher('/servos/targets', Servos, queue_size=10)
        self.buoy_hitter = VizServo(self.p)
        msg = Servos()
        msg.targets[0] = 0
        self.servo_pub.publish(msg)
        self.dropped = 0

    def run(self) -> None:
        # self.p.status("retare!")

        # self.mu.retare(START_X, START_Y, 0)

        self.p.status("hello! arming the sub.")
        self.mu.enable()

        # Dive
        self.p.status("Diving")
        self.dive.run(1)

        # Gate
        self.p.status("Gate setup")
        gate_offset_x = 1.5*cos(GATE_ANGLE)
        gate_offset_y = 1.5*sin(GATE_ANGLE)
        self.mu.goto((GATE_POS_X - gate_offset_x, GATE_POS_Y - gate_offset_y, .7), heading=GATE_ANGLE)

        self.p.status("Gate spin")
        self.mu.goto_relative((0,0,.7), heading=4*3.14159265897932, v=1)

        self.p.status("Gate - go! go!")
        self.mu.goto((GATE_POS_X + gate_offset_x, GATE_POS_Y + gate_offset_y, .7), heading=GATE_ANGLE)
        self.p.status("Gate done!")

        # Marker
        self.p.status("I hate the marker dropper")
        self.mu.goto((MARKER_X, MARKER_Y, 1.5), 0)
        self.p.status("zoinks")
        self.drop()
        self.p.status("jeepers")
        self.drop()
        
        # Octagon
        self.p.status("ascending a bit")
        self.mu.goto_relative((0,0,.7))
        self.p.status("WERE GONNA HIT THE PENTAGON")
        self.mu.goto((OCT_POS_X, OCT_POS_Y, .7), v=0.7)
        self.p.status("BOOM")
        self.mu.goto_relative((0,0,-0.1))
        self.mu.disable()
        rospy.sleep(10)
        self.mu.enable()
        self.mu.goto_relative((0,0,0.7))


        # Buoy
        self.p.status("baba buoy!")
        self.mu.goto((BUOY_SEARCH_X1, BUOY_SEARCH_Y1, BUOY_SEARCH_DEPTH), heading=BUOY_LOOK_ANGLE, v=0.7)
        self.mu.goto((BUOY_SEARCH_X2, BUOY_SEARCH_Y2, BUOY_SEARCH_DEPTH), heading=BUOY_LOOK_ANGLE)
        # self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MAX, BUOY_SEARCH_DEPTH), heading=BUOY_LOOK_ANGLE)
        self.buoy_hitter.run(BUOYTAG, (BUOY_X, BUOY_Y, 2.5))

        # disarm
        self.p.status("done")
        self.mu.disable()

        start_pos = self.mu.get_position()
        # self.square.run(2)
        pos = self.mu.get_position()
        # self.mu.goto()

    
    def drop(self):
        pos_neutral = 0
        pos_drop0 = -90
        pos_drop1 = 90

        msg = Servos()

        if self.dropped == 0:
            msg.targets[0] = pos_drop0
            self.servo_pub.publish(msg)
            rospy.sleep(2)
            msg.targets[0] = pos_neutral
            self.servo_pub.publish(msg)
            self.dropped += 1
            return

        if self.dropped == 1:
            msg.targets[0] = pos_drop1
            self.servo_pub.publish(msg)
            rospy.sleep(2)
            msg.targets[0] = pos_neutral
            self.servo_pub.publish(msg)
            self.dropped += 1
            return

    def cancel(self) -> None:
        self.p.status("byeeee")
        self.dive.cancel()
        self.mu.disable()
        # self.square.cancel()
