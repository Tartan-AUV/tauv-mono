import rospy
from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
from threading import Event


@dataclass
class TaskResources:
    pass


TaskStatus = IntEnum


@dataclass
class TaskResult:
    status: TaskStatus


class Task(ABC):

    def __init__(self) -> None:
        self._cancel_event: Event = Event()

    @abstractmethod
    def run(self, resources: TaskResources) -> TaskResult:
        pass

    @abstractmethod
    def _handle_cancel(self, resources: TaskResources):
        pass

    def cancel(self):
        rospy.logdebug('cancel setting cancel_event')
        self._cancel_event.set()

    def _check_cancel(self, resources: TaskResources):
        rospy.logdebug('_check_cancel checking cancel_event')
        if self._cancel_event.is_set():
            rospy.logdebug('_check_cancel cancel_event is set, running _handle_cancel')
            self._cancel_event.clear()

            self._handle_cancel(resources)
