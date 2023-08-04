from typing import Optional
from spatialmath import SE3, SO3, SE2
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, goto, gate, detect_pinger
from enum import IntEnum


class KFTransdec23State(IntEnum):
    UNKNOWN = 0
    DIVE = 1
    GOTO_GATE = 2
    GATE = 3
    PINGER = 4

class KFTransdec23(Mission):

    def __init__(self):
        self._state = KFTransdec23State.UNKNOWN

        self._gate_pose: SE3 = SE2(3, 0, 0)

    def entrypoint(self) -> Optional[Task]:
        self._state = KFTransdec23State.DIVE
        return dive.Dive(delay=20.0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == KFTransdec23State.DIVE \
            and task_result.status == dive.DiveStatus.SUCCESS:
            self._state = KFTransdec23State.GOTO_GATE
            return goto.Goto(self._gate_pose, in_course=True)
        elif self._state == KFTransdec23State.GOTO_GATE \
            and task_result.status == goto.GotoStatus.SUCCESS:
            self._state = KFTransdec23State.GATE
            return gate.Gate()
        elif self._state == KFTransdec23State.GATE:
            if task_result.status == gate.GateStatus.GATE_NOT_FOUND:
                return gate.Gate()
            elif task_result.status == gate.GateStatus.SUCCESS:
                return detect_pinger.DetectPinger(30000, 2)
        else:
            return None