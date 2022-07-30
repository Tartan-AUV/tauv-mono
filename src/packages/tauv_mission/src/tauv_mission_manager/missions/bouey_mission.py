from tauv_mission_manager.mission_utils import Mission, Task, TaskParams
from tauv_mission_manager.tasks import Dive, Bouey

class BuoyMission(Mission):
    x = 3
    def __init__(self, params: TaskParams) -> None:
        self.p = params
        self.p.status("I'm being born :O")

        self.dive = Dive(params)
        self.bouey = Bouey(params)

    def run(self) -> None:
        self.p.status("hello")
        self.dive.run(1)
        self.bouey.run("badge")

    def cancel(self) -> None:
        self.p.status("byeeee")
        self.dive.cancel()
        self.bouey.cancel()
