from motion.trajectories.trajectories import TrajectoryStatus
from tauv_mission_manager.mission_utils import Mission, Task, TaskParams
from tauv_mission_manager.tasks import Dive, Square, VizServo, VizServoXY
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

MARKERTAG = "PHONE"

class MarkerMission(Mission):
    x = 3
    def __init__(self, params: TaskParams) -> None:
        self.p: TaskParams = params
        self.mu = params.mu
        self.p.status("I'm being born :O")
        self.dive = Dive(params)
        self.finder = Finder()

        self.marker_hitter = VizServoXY(self.p, drop_height=0.3, heading=0)

    def run(self) -> None:
        # self.p.status("retare!")

        # self.mu.retare(START_X, START_Y, 0)

        self.p.status("hello! arming the sub.")
        self.mu.enable()

        # Dive
        self.p.status("Diving")
        self.dive.run(0.5)

        # Buoy
        # self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MIN, BUOY_SEARCH_DEPTH))
        # self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MAX, BUOY_SEARCH_DEPTH))
        self.marker_hitter.run(MARKERTAG, (1, 1, 1))

        # disarm
        self.p.status("done")
        self.mu.disable()

        # start_pos = self.mu.get_position()
        # self.square.run(2)
        # pos = self.mu.get_position()
        # self.mu.goto()

    def cancel(self) -> None:
        self.p.status("byeeee")
        self.dive.cancel()
        self.mu.disable()
        self.square.cancel()
