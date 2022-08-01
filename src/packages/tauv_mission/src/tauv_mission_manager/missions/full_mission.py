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

POOL_LEN = 25 * YD - 1 * M
LANE_WIDTH_Y = 7 * FT
LANE_WIDTH_X = 2.75 * M

# OCT_POS_X = POOL_LEN - (.5 + .67) * LANE_WIDTH_X
# OCT_POS_Y = 2.75 * LANE_WIDTH_Y

# GATE_POS_X = 1 * LANE_WIDTH_X
# GATE_POS_Y = 2.7 * LANE_WIDTH_Y

OCT_POS_X = POOL_LEN - (.5 + .67) * LANE_WIDTH_X - 2 * M
OCT_POS_Y = 2.6 * LANE_WIDTH_Y + .5 * M

GATE_POS_X = 1 * LANE_WIDTH_X
GATE_POS_Y = 0.5 * LANE_WIDTH_Y


BUOY_SEARCH_X = 1.6 * LANE_WIDTH_X
BUOY_SEARCH_Y_MIN = 3 * LANE_WIDTH_Y
BUOY_SEARCH_Y_MAX = 5 * LANE_WIDTH_Y
BUOY_SEARCH_DEPTH = 2
BUOY_LOOK_ANGLE = 3.14
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
        self.mu.goto_relative((0,0,1))
        self.p.status("WERE GONNA BLOW UP THE OCTAGON")
        self.mu.goto((OCT_POS_X, OCT_POS_Y, 1))
        self.mu.goto_relative((0,0,-0.1))

        # Buoy
        self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MIN, BUOY_SEARCH_DEPTH))
        self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MAX, BUOY_SEARCH_DEPTH))
        self.buoy_hitter.run(BUOYTAG)

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
