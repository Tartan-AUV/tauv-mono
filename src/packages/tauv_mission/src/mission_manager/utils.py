from abc import ABCMeta, abstractclassmethod
from dataclasses import dataclass
from enum import Enum

from motion.motion_utils import MotionUtils

@dataclass
class TaskParams:
    mu: MotionUtils
    # will add vision utils and alarm system when they are ready

class Task(metaclass=ABCMeta):
    @abstractclassmethod
    def __init__(self, params: TaskParams, **kwargs) -> None:
        pass

    @abstractclassmethod
    def run(self) -> Enum:
        pass

class Mission(metaclass=ABCMeta):
    @abstractclassmethod
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def get_finish_state(self) -> Enum:
        pass
