from abc import abstractclassmethod
from enum import Enum
import typing

class FailureLevel(Enum):
    NO_FAILURE = 0
    PREDIVE_FAILURE = 1
    MISSION_FAILURE = 2
    CRITICAL_FAILURE = 3


class AlarmType:
    def __init__(self,
            name: str,
            id: int,
            failure_level: FailureLevel,
            default_set: bool,
            description: str,
            author: str):
        self.name = name
        self.id = id
        self.description = description
        self.author = author
        self.failure_level = failure_level
        self.default_set = default_set

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        return False

    def __hash__(self):
        return self.id

    def __str__(self):
        return f"Alarm {self.id}: {self.name}"


# class AlarmMask():
#     def __init__(self):
#         self._alarms: typing.Set[AlarmType] = set()

#     def set(self, alarm: AlarmType, set: bool=True):
#         if set:
#             self.alarms.add(alarm)
#         else:
#             self.alarms.remove(alarm)

#     def clear(self, alarm: AlarmType):
#         self.set(alarm, False)
    
#     def __eq__(self, __o: object) -> bool:
#         if isinstance(__o, self.__class__):
#             return self._alarms == __o._alarms
#         return False


# Metaclass that allows for the enum-like behavior of the Alarm class.
class AlarmMeta(type):
    def __call__(cls, id: int):
        # TODO: can we do this without creating a new idhash each time?
        return cls.__idhash__[id]

    def __new__(mcls, name, bases, namespace, **kw):
        cls =  super().__new__(mcls, name, bases, namespace, **kw)
        cls.__idhash__ = {a.id : a for a in cls}
        return cls

    def __iter__(cls):
        yield from [cls.__dict__[k] for k in cls.__dict__.keys() if k[:2] != '__']

