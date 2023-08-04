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
        self._world_t_initial : Optional[SE3] = None

    def entrypoint(self) -> Optional[Task]:
        return tasks.start.GetPose()

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if isinstance(task, tasks.start.GetPose) and task_result.status == tasks.start.GetPoseStatus.SUCCESS:
            self._world_t_initial = task_result.pose
            return tasks.dive.Dive(1.0)
        elif isinstance(task, tasks.dive.Dive) and task_result.status == tasks.dive.DiveStatus.SUCCESS:
            return tasks.find.Find("gate", self._params["find"])
        else:
            return None