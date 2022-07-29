from genericpath import isfile
from xmlrpc.client import SERVER_ERROR
from motion.motion_utils import MotionUtils
import rospy
from tauv_alarms.alarm_client import AlarmClient, Alarm
from tauv_messages.messager import Messager
from tauv_mission_manager.mission_utils import Task, TaskParams, Mission
from tauv_msgs.srv import RunBasicMission
import typing
from std_srvs.srv import Trigger
import importlib
import os
import glob
import tauv_mission_manager.missions

class MissionManager:
    def __init__(self) -> None:
        # self.mu = MotionUtils()
        self.ac = AlarmClient()
        self.mu = None

        self.active_mission: typing.Optional[Mission] = None
        self.log = Messager("Mission", color=207)

        rospy.Service('cancel_mission', Trigger, self.cancel)

        rospy.Service('start_mission/square', RunBasicMission, lambda x: self.run_mission(x, 'square'))
        rospy.Service('start_mission/coin', RunBasicMission, lambda x: self.run_mission(x, 'coin'))

        self.ac.clear(Alarm.MISSION_MANAGER_NOT_INITIALIZED, "Initialized!")

    def run_mission(self, srv: RunBasicMission._request_class, name: str):
        if self.active_mission is not None:
            res = RunBasicMission._response_class()
            res.success = False
            res.message = "Mission already running!"

        taskparams = TaskParams(
            status=lambda x: self.log.log(f"task: {x}"),
            mu=self.mu,
            ac=self.ac
        )
        
        mission_type = self.get_mission(name)
        self.active_mission = mission_type(taskparams)
        rospy.Timer(rospy.Duration(max(srv.delay, 1.0)), self.do_start_mission, oneshot=True)
        self.log.log(f"Starting mission in {srv.delay:.1f} seconds...")

    def do_start_mission(self, timer_event):
        if self.active_mission is None:
            self.log.log("Failed to start mission :(", severity=Messager.SEV_ERROR)
            return
        self.active_mission.run()
        self.log.log("Mission complete!")

    def cancel(self, srv: Trigger._request_class):
        if self.active_mission is not None:
            self.active_mission.cancel()
            self.log.log("Cancelled a mission.")
        else:
            self.log.log("No active mission to cancel, doing nothing.")
        self.active_mission = None

    def get_mission(self, name: str) -> typing.Type[Mission]:
        print("reloading missions:")

        mapping = {}

        # TODO: this is kind of stupid and the string->type matching is stupid
        importlib.reload(tauv_mission_manager.missions.square_mission)
        from tauv_mission_manager.missions.square_mission import SquareMission
        mapping['square'] = SquareMission

        importlib.reload(tauv_mission_manager.missions.coin_mission)
        from tauv_mission_manager.missions.square_mission import CoinMission
        mapping['coin'] = CoinMission

        print(SquareMission.x)

        return mapping[name]

def main():
    rospy.init_node('mission_manager')
    MissionManager()
    rospy.spin()
