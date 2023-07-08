from typing import Dict, Optional
from mission_manager.mission import Mission
from mission_manager.task import Task
import mission_manager.tasks as tasks


class KFTransdec23(Mission):

    def __init__(self):
        pass

    def entrypoint(self) -> Optional[Task]:
        return tasks.Dive()

    def transition(self, task: Task, task_result: Task.Result) -> Optional[Task]:
        if isinstance(task, tasks.Dive) and task_result == tasks.Dive.Result.SUCCESS:
            return tasks.Goto()
        else:
            return None
