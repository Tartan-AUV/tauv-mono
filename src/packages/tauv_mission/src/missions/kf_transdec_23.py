from typing import Optional, Dict
from spatialmath import SE3, SO3, SE2
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, goto, gate, detect_pinger, start, find, dive, goto_tag
from enum import IntEnum


class KFTransdec23State(IntEnum):
    START = 0
    DIVE = 1
    GOTO_GATE = 2
    GATE = 3
    PINGER = 4
    UNKOWN = 5

class KFTransdec23(Mission):
    def __init__(self, params : Dict, course_info : Dict):
        self._state = KFTransdec23State.UNKNOWN
        
        self._params : Dict = params
        self._course_info : Dict = course_info

    def entrypoint(self) -> Optional[Task]:
        self._status = KFTransdec23State.START
        return start.Start()

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state==KFTransdec23State.START \
            and task_result.status==start.StartStatus.SUCCESS:
            self._state = KFTransdec23State.DIVE
            return dive.Dive(1.0)
        elif self._state==KFTransdec23State.DIVE \
            and task_result.status == dive.DiveStatus.SUCCESS:
            self._state = KFTransdec23State.GOTO_GATE
            return find.Find("gate", self._course_info["gate"], self._params["find"])
        elif self._state==KFTransdec23State.GOTO_GATE \
            and task_result.status==find.FindStatus.FOUND:
            self._state = KFTransdec23State.GATE
            pose = SE3.Rt(SO3(), t=[1.0, 0.0, 0.0])
            return goto.GotoTag("gate", pose)
        elif self._state==KFTransdec23State.GATE \
            and task_result.status==find.FindStatus.SUCCESS:
            return detect_pinger.DetectPinger(30000, 2)
        else:
            return None