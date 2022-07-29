import rospy
from abc import ABCMeta, abstractclassmethod, abstractmethod
from dataclasses import dataclass
import typing
from motion.motion_utils import MotionUtils
from tauv_alarms.alarm_client import Alarm, AlarmClient

@dataclass
class TaskParams:
    status: typing.Callable[[str], None]
    mu: MotionUtils
    ac: AlarmClient

class Task(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, params: TaskParams) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def cancel(self) -> None:
        pass

# Mission is made of tasks but has the same interface
class Mission(Task):
    pass