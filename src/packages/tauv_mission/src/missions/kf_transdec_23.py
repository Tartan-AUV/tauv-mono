from typing import Optional, Dict, Union
from spatialmath import SE3, SO3
from missions.mission import Mission
from tasks.task import Task, TaskResult
import tasks.start
import tasks.find
import tasks.dive

class KFTransdec23(Mission):
    def __init__(self, params : Dict[str, Union[int, float]]):
        self._params : Dict[Union[int, float]] = params

    def entrypoint(self) -> Optional[Task]:
        return tasks.dive.Dive(1.0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if isinstance(task, tasks.dive.Dive) and task_result.status == tasks.dive.DiveStatus.SUCCESS:
            return tasks.find.Find("gate", self._params["find"])
        else:
            return None