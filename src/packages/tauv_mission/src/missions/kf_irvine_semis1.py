from typing import Optional
from spatialmath import SE3, SO3, SE2
from math import pi
from missions.mission import Mission
from tasks.task import Task, TaskResult
from tasks import dive, goto, gate, scan_rotate, yaw_roll, buoy_24
from enum import IntEnum

class State(IntEnum):
    START = 0
    DIVE = 1
    ROLL = 2
    GOTO_GATE = 3
    GATE = 4
    GOTO_BUOY = 5
    BUOY = 6
    GOTO_OCTAGON = 7
    OCTAGON = 8

class KFIrvineSemis1(Mission):

    def __init__(self):
        self._state = State.START

        self._gate_pose: SE3 = SE3.Rt(SO3(), (4, -1, 0.8)) # center is top of gate
        # self._buoy_pose: SE3 = SE3.Rt(SO3(), (7, 3, 1))
        self._octagon_pose: SE3 = SE3.Rt(SO3(), (15, 0, 0))

        self._buoy_waypoint: SE3 = SE3.Rt(SO3.Rz(-pi/4), (5, 0, 1.5))
        
        self._goto_gate_offset: SE3 = SE3.Rt(SO3(), (-1, 0, 1))
        self._after_gate_offset: SE3 = SE3.Rt(SO3(), (1.0, 0, 1))

        self._octagon_before_offset: SE3 = SE3.Rt(SO3(), (0, 0, 1))

    def entrypoint(self) -> Optional[Task]:
        self._state = State.DIVE
        return dive.Dive(2.0, 0.3, 0.0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if self._state == State.DIVE:
            if task_result.status == dive.DiveStatus.SUCCESS:
                self._state = State.ROLL
                return yaw_roll.YawRoll()

        elif self._state == State.ROLL:
            if task_result.status == yaw_roll.YawRollStatus.SUCCESS:
                self._state = State.GOTO_GATE
                return goto.Goto(self._goto_gate_offset * self._gate_pose)

        elif self._state == State.GOTO_GATE:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.GATE
                return goto.Goto(self._after_gate_offset * self._gate_pose)

        elif self._state == State.GATE:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.GOTO_BUOY
                return goto.Goto(self._buoy_waypoint)

        elif self._state == State.GOTO_BUOY:
            if task_result.status == goto.GotoStatus.SUCCESS:
                self._state = State.BUOY
                return buoy_24.CircleBuoy()

        elif self._state == State.BUOY:
            if task_result.status == buoy_24.CircleBuoyStatus.SUCCESS:
                self._state = State.GOTO_OCTAGON
                return goto.Goto(self._octagon_before_offset * self._octagon_pose)
            
        elif self._state == State.GOTO_OCTAGON:
            if task_result.status == buoy_24.GotoStatus.SUCCESS:
                self._state = State.OCTAGON
                return None
        else:
            return None


