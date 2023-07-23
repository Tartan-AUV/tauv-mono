from typing import Optional
from spatialmath import SE3, SO3
from missions.mission import Mission
from tasks.task import Task, TaskResult
import tasks


class KFTransdec23(Mission):

    def __init__(self):
        pass

    def entrypoint(self) -> Optional[Task]:
        return tasks.dive.Dive(depth=1.0)

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if isinstance(task, tasks.dive.Dive) and task_result.status == tasks.dive.DiveStatus.SUCCESS:
            return tasks.goto.Goto(pose=SE3.Rt(SO3(), (5.0, 0.0, 1.0)))
        else:
            return None
