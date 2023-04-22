from pydoc import locate
import rospy
from tauv_msgs.srv import LoadMission, LoadMissionRequest, LoadMissionResponse

class MissionManager():
    def __init__(self) -> None:
        rospy.Service('load_mission', LoadMission, self.load_mission)
    
    def load_mission(self, req: LoadMissionRequest):
        mission = req.mission
        loaded_mission = locate(f'missions.{mission.lower()}.{mission}')
        if loaded_mission == None:
            return LoadMissionResponse(success=False)

        self.mission = loaded_mission()
        self.finish = self.mission.get_finish_state()
        self.runMission()
        return LoadMissionResponse(success=True)

    def start(self):
        rospy.spin()
    
    def runMission(self):
        state = self.mission.start_state
        while state != self.finish:
            print(state)
            status = self.mission.tasks[state].run()
            state = self.mission.transitions[state](status)

def main():
    rospy.init_node('mission_manager')
    n = MissionManager()
    n.start()