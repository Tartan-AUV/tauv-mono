from motion.trajectories.trajectories import TrajectoryStatus
from tauv_mission_manager.mission_utils import Mission, Task, TaskParams
from tauv_mission_manager.tasks import Dive, Square
from vision.detectors.finder import Finder
import rospy
from tauv_msgs.msg import BucketDetection
from math import sin, cos

M = 1
YD = 0.9144
FT = 0.3048

POOL_LEN = 25 * YD
LANE_WIDTH_X = 7 * FT
LANE_WIDTH_Y = 2.75 * M

OCT_POS_X = 1.5 * LANE_WIDTH_X
OCT_POS_Y = POOL_LEN - (.5 + .67)*LANE_WIDTH_Y

GATE_POS_X = 1.5 * LANE_WIDTH_X
GATE_POS_Y = LANE_WIDTH_Y

START_X = 0.5 * LANE_WIDTH_X
START_Y = 1.4 * LANE_WIDTH_Y

BUOY_SEARCH_X = 1.5 * LANE_WIDTH_X
BUOY_SEARCH_Y_MIN = 2 * LANE_WIDTH_Y
BUOY_SEARCH_Y_MAX = 4 * LANE_WIDTH_Y
BUOY_SEARCH_DEPTH_1 = 3
BUOY_SEARCH_DEPTH_2 = 1.5
BUOY_LOOK_ANGLE = 0
BUOYTAG = "badge"

BUOY_HIT_DIST = 1
BUOY_ATTACK_ANGLE = 0

class PoolMission(Mission):
    x = 3
    def __init__(self, params: TaskParams) -> None:
        self.p: TaskParams = params
        self.mu = params.mu
        self.p.status("I'm being born :O")
        self.dive = Dive(params)
        self.finder = Finder()

    def run(self) -> None:
        self.p.status("hello! arming the sub.")
        self.mu.enable()

        # Dive
        self.p.status("Diving")
        self.dive.run(1)

        # Gate
        self.p.status("Gate setup")
        self.mu.goto((GATE_POS_X - 1.5*M, GATE_POS_Y, 1), heading=0)

        self.p.status("Gate spin")
        self.mu.goto_relative((0,0,1), heading=2*3.14159265897932)

        self.p.status("Gate - go! go!")
        self.mu.goto((GATE_POS_X + 1.5*M, GATE_POS_Y, 1), heading=0)
        self.p.status("Gate done!")

        # Buoy
        self.p.status("baba buoy, where are you???")
        self.mu.goto_relative((0,0,1), heading=2*3.14159265897932)
        
        if self.finder.find_by_tag(BUOYTAG) is None:
            self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MIN, BUOY_SEARCH_DEPTH_1), heading=BUOY_LOOK_ANGLE, block=TrajectoryStatus.EXECUTING)
            while self.mu.get_motion_status().value < TrajectoryStatus.FINISHED.value and \
                    self.finder.find_by_tag(BUOYTAG) is not None:
                rospy.sleep(rospy.Duration(0.2))
        
        if self.finder.find_by_tag(BUOYTAG) is None:
            self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MAX, BUOY_SEARCH_DEPTH_1), heading=BUOY_LOOK_ANGLE, block=TrajectoryStatus.EXECUTING)
            while self.mu.get_motion_status().value < TrajectoryStatus.FINISHED.value and \
                    self.finder.find_by_tag(BUOYTAG) is not None:
                rospy.sleep(rospy.Duration(0.2))
        
        if self.finder.find_by_tag(BUOYTAG) is None:
            self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MAX, BUOY_SEARCH_DEPTH_2), heading=BUOY_LOOK_ANGLE, block=TrajectoryStatus.EXECUTING)
            while self.mu.get_motion_status().value < TrajectoryStatus.FINISHED.value and \
                    self.finder.find_by_tag(BUOYTAG) is not None:
                rospy.sleep(rospy.Duration(0.2))
        
        if self.finder.find_by_tag(BUOYTAG) is None:
            self.mu.goto((BUOY_SEARCH_X, BUOY_SEARCH_Y_MIN, BUOY_SEARCH_DEPTH_2), heading=BUOY_LOOK_ANGLE, block=TrajectoryStatus.EXECUTING)
            while self.mu.get_motion_status().value < TrajectoryStatus.FINISHED.value and \
                    self.finder.find_by_tag(BUOYTAG) is not None:
                rospy.sleep(rospy.Duration(0.2))

        if self.finder.find_by_tag(BUOYTAG) is None:
            self.p.status("no buoy :(")
        else:
            self.p.status("Found the buoy!")
            self.p.status("Hunting buoy for 20 seconds:")
            t0 = rospy.Time.now()
            while not rospy.Time.now() - t0 <= rospy.Duration(20):
                det :BucketDetection  = self.finder.find_by_tag(BUOYTAG)
                bx = det.position.x
                by = det.position.y
                bz = det.position.z
                dist_x = abs(bx - BUOY_SEARCH_X)
                self.mu.goto((BUOY_SEARCH_X, by + dist_x*sin(BUOY_ATTACK_ANGLE), bz), heading=BUOY_ATTACK_ANGLE, block=TrajectoryStatus.EXECUTING)
                rospy.sleep(1)

            self.p.status("Approaching buoy for 10 seconds:")
            t0 = rospy.Time.now()
            bz = 0
            while not rospy.Time.now() - t0 <= rospy.Duration(10):
                det :BucketDetection  = self.finder.find_by_tag(BUOYTAG)
                bx = det.position.x
                by = det.position.y
                bz = det.position.z

                prehit_pos_x = bx - BUOY_HIT_DIST * cos(BUOY_ATTACK_ANGLE)
                prehit_pos_y = by - BUOY_HIT_DIST * sin(BUOY_ATTACK_ANGLE)
                prehit_pos_z = bz
                self.mu.goto((prehit_pos_x, prehit_pos_y, prehit_pos_z), heading=BUOY_ATTACK_ANGLE, block=TrajectoryStatus.EXECUTING)
                rospy.sleep(1)

            self.p.status("bonk da buoy")
            self.mu.goto_relative((BUOY_HIT_DIST, 0, bz), heading=0)
            self.p.status("bonk!")
            self.mu.goto_relative((-BUOY_HIT_DIST, 0, bz), heading=0)

            self.p.status("buoy bonk done!")

            
            self.mu.goto()

        # Octagon
        self.p.status("ascending a bit")
        self.mu.goto_relative((0,0,1))
        self.p.status("WERE GONNA BLOW UP THE OCTAGON")
        self.mu.goto((OCT_POS_X, OCT_POS_Y, 1))
        self.mu.goto_relative(0,0,-0.1)

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
        self.square.cancel()
