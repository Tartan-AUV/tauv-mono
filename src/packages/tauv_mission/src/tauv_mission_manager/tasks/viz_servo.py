from importlib.abc import Finder
from motion.trajectories.trajectories import TrajectoryStatus
import rospy
from tauv_mission_manager.mission_utils import Task, TaskParams
from vision.detectors.finder import Finder
import typing
from tauv_msgs.msg import BucketDetection
import numpy as np
from tauv_util.types import tl

class VizServo(Task):
    def __init__(self, params: TaskParams) -> None:
        self.mu = params.mu
        self.cancelled = False
        self.status = params.status
        self.f = Finder()

    def run(self, tag, nominal_pos):
        if (self.cancelled): return
        
        self.status(f"Looking for a {tag}")
        hunting = True
        self.status("hunting!")
        t0 = rospy.Time.now()
        while rospy.Time.now() - t0 < rospy.Duration(60):
            dets: typing.List[BucketDetection] = self.f.find_by_tag(tag)
            # Choose the right buouy

            best_score = 0
            best_pos = None # lowest wins
            for d in dets:
                num_dets = max(d.count, 50)
                if num_dets < 3:
                    continue
                dist = np.sqrt((d.position.x - nominal_pos[0])**2 + (d.position.y - nominal_pos[1])**2)
                score = dist - num_dets/20
                if score < best_score:
                    best_score = score
                    best_pos = d.position

            if best_pos is None:
                self.status("Could not find object.")
                return

            my_pos = self.mu.get_position()
            vec_to_obj = np.array(best_pos.x, best_pos.y, best_pos.z) - my_pos
            

            if np.linalg.norm(vec_to_obj) > 0.25:
                self.mu.goto(tl(best_pos), block=TrajectoryStatus.EXECUTING)
            else:
                self.status("Bop!")
                self.mu.goto(tl(best_pos), block=TrajectoryStatus.FINISHED)
                self.mu.goto_relative((-1, 0, best_pos.z))
                return

            rospy.sleep(0.5)

        self.status("Timed out :/")
        return

    def cancel(self):
        self.mu.abort()
        self.cancelled = True
