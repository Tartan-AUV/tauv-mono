from tauv_mission_manager.mission_utils import Mission, Task, TaskParams

class SquareMission(Mission):
    x = 3
    def __init__(self, params: TaskParams) -> None:
        self.p = params
        self.p.status("I'm being born :O")

    def run(self) -> None:
        self.p.status("hello")

    def cancel(self) -> None:
        self.p.status("byeeee")
