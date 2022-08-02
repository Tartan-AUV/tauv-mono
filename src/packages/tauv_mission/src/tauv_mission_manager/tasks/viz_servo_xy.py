from importlib.abc import Finder
from motion.trajectories.trajectories import TrajectoryStatus
import rospy
from tauv_mission_manager.mission_utils import Task, TaskParams
from vision.detectors.finder import Finder
import typing
from tauv_msgs.msg import BucketDetection, Servos
import numpy as np
from tauv_util.types import tl

class VizServo(Task):
    APROACHING = 0
    SERVOING = 1
    DONE = 2

    def __init__(self, params: TaskParams, drop_height=0.3, heading=0) -> None:
        self.mu = params.mu
        self.cancelled = False
        self.status = params.status
        self.f = Finder()
        self.servo_pub = rospy.Publisher('/thrusters/servos', Servos, queue_size=10)
        self.dropped = 0
        self.current_height = 0
        self.drop_height = 0.3
        self.drop_heading = heading

    def run(self, tag, nominal_pos):
        if (self.cancelled): return
        self.state = VizServo.APROACHING
        
        self.status(f"Looking for a {tag}")
        hunting = True
        self.status("hunting!")
        t0 = rospy.Time.now()
        while rospy.Time.now() - t0 < rospy.Duration(60) and self.state != VizServo.DONE:
            dets: typing.List[BucketDetection] = self.f.find_by_tag(tag)
            # Choose the right buouy

            best_score = 0
            best_pos = None # lowest wins
            for d in dets:
                num_dets = max(d.count, 50)
                if num_dets < 3:
                    self.status(f"found a bad detection")
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
            vec_to_obj_xy = np.array([best_pos.x, best_pos.y]) - my_pos[0:2]
            vec_to_obj_xy_len = np.linalg.norm(vec_to_obj_xy)

            if self.state == VizServo.APROACHING:
                if vec_to_obj_xy_len > 1:
                    self.mu.goto(tl(best_pos[0], best_pos[1], 0.5), heading=self.drop_heading, block=TrajectoryStatus.EXECUTING)
                else:
                    self.mu.goto(tl(best_pos[0], best_pos[1], 0.5), heading=self.drop_heading)
                    self.state = VizServo.SERVOING
                    self.status("Switching to servoing!")
                    self.current_height = my_pos[2]
                rospy.sleep(0.1)


            elif self.state == VizServo.SERVOING:
                tgt_height = best_pos.z - self.drop_height

                dt = 0.05
                self.current_height = max(tgt_height, self.current_height - 0.2 * dt)
                self.mu.goto_pid((best_pos[0], best_pos[1], self.current_height), heading=self.drop_heading)

                if self.current_height <= tgt_height and self.dropped < 2:
                    self.status("Bombs away!")
                    self.drop()
                    self.mu.goto_pid(best_pos[0], best_pos[1], 1)

            if self.dropped == 2:
                self.state = VizServo.DONE
                self.status("Done dropping!")
                return

    def drop(self):
        pos_neutral = 0.5
        pos_drop0 = 0.4
        pos_drop1 = 0.6

        msg = Servos()

        if self.dropped == 0:
            msg.targets[0] = pos_drop0
            self.servo_pub.publish(msg)
            rospy.sleep(2)
            msg.targets[0] = pos_neutral
            self.servo_pub.publish(msg)
            self.dropped += 1
            return

        if self.dropped == 1:
            msg.targets[0] = pos_drop1
            self.servo_pub.publish(msg)
            rospy.sleep(2)
            msg.targets[0] = pos_neutral
            self.servo_pub.publish(msg)
            self.dropped += 1
            return


    def cancel(self):
        self.mu.abort()
        self.cancelled = True
