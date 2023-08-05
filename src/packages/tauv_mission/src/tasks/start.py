from spatialmath import SE2, SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class StartStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1

@dataclass
class StartResult(TaskResult):
    status: StartStatus

class Start(Task):
    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources) -> StartResult:
        odom_t_vehicle_initial = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        resources.transforms.set_a_to_b('kf/odom', 'kf/start', odom_t_vehicle_initial)

        return StartResult(StartStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        pass