import time
import rospy
from dataclasses import dataclass
from spatialmath import SE3
from tasks.task import Task, TaskResources, TaskStatus, TaskResult


class GotoTagStatus(TaskStatus):
    SUCCESS = 0
    FAILURE = 1
    CANCELLED = 2 

@dataclass
class GotoTagResult(TaskResult):
    status: GotoTagStatus

class GotoTag(Task):
    def __init__(self, tag : str, trans: SE3):
        super().__init__()
        self._tag : str = tag
        self._trans: SE3 = trans

    def run(self, resources: TaskResources) -> GotoTagResult:
        cur_pose = resources.transforms.get_a_to_b('kf/odom', 'kf/vehicle')
        detection = resources.map.find_closest(self._tag, cur_pose.t)

        if(detection is None):
            return GotoTagResult(GotoTagStatus.FAILURE)
        
        resources.motion.goto(detection.pose*self._trans)

        while True:
            if resources.motion.wait_until_complete(timeout=rospy.Duration.from_sec(0.1)):
                break

            if self._check_cancel(resources): return GotoTagResult(GotoTagStatus.CANCELLED)

        return GotoTagResult(GotoTagStatus.SUCCESS)

    def _handle_cancel(self, resources: TaskResources):
        resources.motion.cancel()