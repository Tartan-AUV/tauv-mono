from tauv_mission_manager.mission_utils import Mission, Task, TaskParams
from tauv_mission_manager.tasks import Dive, Square

M = 1
YD = 0.9144

POOL_LEN = 25 * YD
LANE_WIDTH = 2.5 * YD

OCT_DIST= POOL_LEN - (.5 + .67) * 2.5 * M
OCT_IN = 1.5 * LANE_WIDTH

class FullMission(Mission):
    x = 3
    def __init__(self, params: TaskParams) -> None:
        self.p = params
        self.p.status("I'm being born :O")

        self.dive = Dive(params)

    def run(self) -> None:
        self.p.status("hello")
        self.dive.run(1)
        # self.square.run(2)
        pos = self.mu.get_position()
        p0 = Waypoint(pos)

    def cancel(self) -> None:
        self.p.status("byeeee")
        self.dive.cancel()
        self.square.cancel()
