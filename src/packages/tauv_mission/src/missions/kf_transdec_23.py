from typing import Optional, Dict, Union
from spatialmath import SE3, SO3
from missions.mission import Mission
from tasks.task import Task, TaskResult
import tasks.start
import tasks.find
import tasks.dive
import tasks.goto_tag

class KFTransdec23(Mission):
    def __init__(self, params : Dict, course_info : Dict):
        self._params : Dict = params
        self._course_info : Dict = course_info

    def entrypoint(self) -> Optional[Task]:
        return tasks.start.Start()

    def transition(self, task: Task, task_result: TaskResult) -> Optional[Task]:
        if isinstance(task, tasks.start.Start) and task_result.status==tasks.start.StartStatus.SUCCESS:
            return tasks.dive.Dive(1.0)
        elif isinstance(task, tasks.dive.Dive) and task_result.status == tasks.dive.DiveStatus.SUCCESS:
            return tasks.find.Find("gate", self._course_info["gate"], self._params["find"])
        elif isinstance(task, tasks.find.Find) and task_result.tag=="gate" and task_result.status==tasks.find.FindStatus.SUCCESS:
            pose = SE3.Rt(SO3(), t=[1.0, 0.0, 0.0])
            return tasks.goto.GotoTag("gate", pose)
        else:
            return None