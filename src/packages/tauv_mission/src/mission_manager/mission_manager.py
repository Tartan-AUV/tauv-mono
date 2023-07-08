import rospy
from typing import Optional
from threading import Lock, Event, Condition, Thread
from mission_manager.task import Task, TaskResources, TaskStatus, TaskResult
from mission_manager.mission import Mission
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
import mission_manager.missions as missions
import mission_manager.tasks as tasks


class MissionManager:

    def __init__(self):
        self._mission_lock: Lock = Lock()
        self._mission: Optional[Mission] = None

        self._task: Optional[Task] = None
        self._task_result: Optional[TaskResult] = None

        self._mission_start_event: Event = Event()
        self._task_cancel_event: Event = Event()
        self._task_finish_event: Event = Event()
        self._task_update_condition: Condition = Condition()

        self._run_mission_service: rospy.Service = rospy.Service("mission/run", Trigger, self._handle_run_mission)
        self._cancel_mission_service: rospy.Service = rospy.Service("mission/cancel", Trigger, self._handle_cancel_mission)

        self._run_thread: Optional[Thread] = None
        self._task_thread: Optional[Thread] = None

    def run(self) -> None:
        def run_task():
            result = self._task.run()
            self._task_finish_event.set()
            self._task_update_condition.notify_all()
            self._task_result = result

        def cancel_task():
            self._task.cancel()
            self._task_finish_event.set()
            self._task_update_condition.notify_all()

        def run_manager():
            while True:
                self._mission_lock.acquire()
                if self._mission is None:
                    self._mission_lock.release()
                    self._mission_start_event.wait()
                    self._mission_start_event.clear()

                    self._task = self._mission.entrypoint()
                    self._task_thread = Thread(target=run_task)
                else:
                    self._mission_lock.release()
                    self._task_update_condition.wait()

                    if self._task_finish_event.set():
                        self._task_finish_event.clear()
                        self._task_cancel_event.clear()

                        self._mission_lock.acquire()
                        self._task = self._mission.transition(self._task, self._task_result)
                        self._mission_lock.release()

                        if self._task is None:
                            self._mission_lock.acquire()
                            self._mission = None
                            self._mission_lock.release()
                        else:
                            self._task_thread = Thread(target=run_task)

                    elif self._task_cancel_event.set():
                        self._task_finish_event.clear()
                        self._task_cancel_event.clear()

                        # Terminate self._task_thread

                        self._task_thread = Thread(target=cancel_task)
                        self._task_update_condition.wait()


                # Wait for task complete or cancel condition

                self._mission_lock.release()

        self._run_thread = Thread(target=run_manager)
        rospy.spin()

    def _handle_run_mission(self, req: TriggerRequest) -> TriggerResponse:
        with self._mission_lock:
            res = TriggerResponse()

            if self._mission is not None:
                print("mission in progress")
                return res

            self._mission = missions.KFTransdec23()

            self._mission_start_event.set()

            return res

    def _handle_cancel_mission(self, req: TriggerRequest) -> TriggerResponse:
        with self._mission_lock:
            res = TriggerResponse()

            if self._mission is None:
                print("no mission to cancel")
                return res

            self._mission = None

            return res
