import rospy
from typing import Optional
from threading import Lock, Event, Thread
from tasks.task import Task, TaskResources, TaskStatus, TaskResult
import missions
from missions.mission import Mission
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse


class MissionManager:

    def __init__(self):
        self._lock: Lock = Lock()

        self._mission: Optional[Mission] = None

        self._task: Optional[Task] = None
        self._task_result: Optional[TaskResult] = None

        self._task_resources: TaskResources = TaskResources()

        self._mission_start_event: Event = Event()
        self._mission_cancel_event: Event = Event()
        self._mission_update_event: Event = Event()

        self._task_done_event: Event = Event()
        self._task_update_event: Event = Event()

        self._run_mission_service: rospy.Service = rospy.Service("mission/run", Trigger, self._handle_run_mission)
        self._cancel_mission_service: rospy.Service = rospy.Service("mission/cancel", Trigger, self._handle_cancel_mission)

        self._spin_mission_thread: Optional[Thread] = None
        self._spin_task_thread: Optional[Thread] = None
        self._task_thread: Optional[Thread] = None

    def run(self) -> None:
        self._spin_mission_thread = Thread(target=self._spin_mission, daemon=True)
        self._spin_mission_thread.start()
        self._spin_task_thread = Thread(target=self._spin_task, daemon=True)
        self._spin_task_thread.start()
        rospy.spin()

    def _spin_mission(self):
        while True:
            self._mission_update_event.wait()

            if self._mission_start_event.is_set():
                self._start_mission()
            elif self._mission_cancel_event.is_set():
                self._cancel_mission()

            self._mission_update_event.clear()

    def _spin_task(self):
        while True:
            self._task_update_event.wait()

            if self._task_done_event.is_set():
                self._transition_task()

            self._task_done_event.clear()
            self._task_update_event.clear()

    def _start_mission(self):
        with self._lock:
            self._task = self._mission.entrypoint()
            self._task_thread = Thread(target=self._run_task, daemon=True)
            self._task_thread.start()

            self._mission_start_event.clear()

    def _cancel_mission(self):
        with self._lock:
            self._mission = None

            if self._task is not None:
                self._task.cancel()

    def _transition_task(self):
        with self._lock:
            if self._mission_cancel_event.is_set():
                self._mission_cancel_event.clear()
                return

            new_task = self._mission.transition(self._task, self._task_result)

            if new_task is None:
                self._mission = None
                self._task = None
                self._task_result = None
                return

            self._task = self._mission.transition(self._task, self._task_result)
            self._task_result = None
            self._task_thread = Thread(target=self._run_task, daemon=True)
            self._task_thread.start()

    def _run_task(self):
        task_result = self._task.run(self._task_resources)
        with self._lock:
            self._task_result = task_result
        self._task_done_event.set()
        self._task_update_event.set()

    def _handle_run_mission(self, req: TriggerRequest) -> TriggerResponse:
        with self._lock:
            res = TriggerResponse()

            if self._mission is not None:
                print("mission in progress")
                return res

            self._mission = missions.kf_transdec_23.KFTransdec23()

            self._mission_start_event.set()
            self._mission_update_event.set()

            return res

    def _handle_cancel_mission(self, req: TriggerRequest) -> TriggerResponse:
        with self._lock:
            res = TriggerResponse()

            if self._mission is None:
                print("no mission to cancel")
                return res

            self._mission_cancel_event.set()
            self._mission_update_event.set()

            return res


def main():
    rospy.init_node('mission_manager')
    node = MissionManager()
    node.run()