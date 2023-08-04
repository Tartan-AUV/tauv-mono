from spatialmath import SE2, SE3, SO3
from dataclasses import dataclass
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class GetPoseStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1

@dataclass
class GetPoseResult(TaskResult):
    status: GetPoseStatus
    pose: SE3

class GetPose(Task):
    def __init__(self):
        super().__init__()

    def run(self, resources: TaskResources) -> GetPoseResult:
        odom_t_vehicle_initial = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')

        return GetPoseResult(GetPoseStatus.SUCCESS, odom_t_vehicle_initial)

    def _handle_cancel(self, resources: TaskResources):
        pass