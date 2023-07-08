from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
from typing import Callable
from threading import Event

from motion_client.motion_client import MotionClient
from map_client.map_client import MapClient


@dataclass
class TaskResources:
    motion: MotionClient
    map: MapClient


TaskStatus = IntEnum


@dataclass
class TaskResult:
    status: TaskStatus


class Task(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self, resources: TaskResources) -> TaskResult:
        pass

    @abstractmethod
    def cancel(self, resources: TaskResources):
        pass
