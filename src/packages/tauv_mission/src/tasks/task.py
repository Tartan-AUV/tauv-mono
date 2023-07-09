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
        self._cancel_event.set()

    def _check_cancel(self, resources: TaskResources):
        if self._cancel_event.is_set():
            self._cancel_event.clear()

            self._handle_cancel(resources)
