from motion.trajectories.trajectories import TrajectoryStatus
from tauv_mission_manager.mission_utils import Mission, Task, TaskParams
from tauv_mission_manager.tasks import Dive, Square, VizServo
from vision.detectors.finder import Finder
import rospy
from tauv_msgs.msg import BucketDetection
from math import sin, cos

M = 1
YD = 0.9144
FT = 0.3048
IN = FT / 12

# POOL_LEN = 25 * YD - 1 * M
LANE_WIDTH_Y = 7 * FT
LANE_WIDTH_X = 9 * FT
POOL_LEN = 7*LANE_WIDTH_X + 6*2*FT

# OCT_POS_X = POOL_LEN - (.5 + .67) * LANE_WIDTH_X
# OCT_POS_Y = 2.75 * LANE_WIDTH_Y

# GATE_POS_X = 1 * LANE_WIDTH_X
# GATE_POS_Y = 2.7 * LANE_WIDTH_Y

# OCT_POS_X = POOL_LEN - (.5 + .67) * LANE_WIDTH_X - 2 * M
OCT_POS_X = 6.1 * LANE_WIDTH_X + 0.1 * M
OCT_POS_Y = 4.2 * LANE_WIDTH_Y + 0.25 * M

GATE_POS_X = 1 * LANE_WIDTH_X
GATE_POS_Y = 2.65 * LANE_WIDTH_Y

BUOY_SEARCH_X1 = 5.0 * LANE_WIDTH_X
BUOY_SEARCH_Y1 = 0.5 * LANE_WIDTH_Y
BUOY_SEARCH_X2 = 4.0 * LANE_WIDTH_X
BUOY_SEARCH_Y2 = 0.5 * LANE_WIDTH_Y
# BUOY_SEARCH_Y_MIN = 3 * LANE_WIDTH_Y
# BUOY_SEARCH_Y_MAX = 5 * LANE_WIDTH_Y
BUOY_SEARCH_DEPTH = 0.7
PI = 3.14
BUOY_LOOK_ANGLE = 3.15*PI/4
BUOYTAG = "badge"

class PoolMission(Mission):
    x = 3
    def __init__(self, params: TaskParams) -> None:
        self.p: TaskParams = params
        self.mu = params.mu
        self.p.status("I'm being born :O")
        self.dive = Dive(params)
        self.finder = Finder()

        self.buoy_hitter = VizServo(self.p)

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
        self.mu.goto((GATE_POS_X - 1.5*M, GATE_POS_Y, .7), heading=0)

        self.p.status("Gate spin")
        self.mu.goto_relative((0,0,.7), heading=4*3.14159265897932, v=1)

        self.p.status("Gate - go! go!")
        self.mu.goto((GATE_POS_X + 1.5*M, GATE_POS_Y, .7), heading=0)
        self.p.status("Gate done!")

        
        # Octagon
        self.p.status("ascending a bit")
        self.mu.goto_relative((0,0,.7))
        self.p.status("WERE GONNA HIT THE PENTAGON")
        self.mu.goto((OCT_POS_X, OCT_POS_Y, .7), v=0.7)
        self.p.status("BOOM")
        self.mu.goto_relative((0,0,-0.5))
        self.mu.disable()
        rospy.sleep(5)
        self.mu.enable()
        self.mu.goto_relative((0,0,0.7))

        # Buoy
        self.p.status("baba buoy!")
        self.mu.goto((BUOY_SEARCH_X1, BUOY_SEARCH_Y1, BUOY_SEARCH_DEPTH), heading=BUOY_LOOK_ANGLE)
        self.mu.goto((BUOY_SEARCH_X2, BUOY_SEARCH_Y2, BUOY_SEARCH_DEPTH), heading=BUOY_LOOK_ANGLE)
        # self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MAX, BUOY_SEARCH_DEPTH), heading=BUOY_LOOK_ANGLE)
        self.buoy_hitter.run(BUOYTAG, (2.5*LANE_WIDTH_X, 1.5*LANE_WIDTH_Y, 2.5))

        # disarm
        self.p.status("done")
        self.mu.disable()

        start_pos = self.mu.get_position()
        # self.square.run(2)
        pos = self.mu.get_position()
        # self.mu.goto()

    def cancel(self) -> None:
        self.p.status("byeeee")
        self.dive.cancel()
        self.mu.disable()
        self.square.cancel()
