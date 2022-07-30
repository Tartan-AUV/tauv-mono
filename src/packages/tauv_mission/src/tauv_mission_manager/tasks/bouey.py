from motion.trajectories.trajectories import TrajectoryStatus
import rospy
from tauv_mission_manager.mission_utils import Task, TaskParams
from motion.trajectories.linear_trajectory import LinearTrajectory, Waypoint
from vision.detectors.finder import Finder

class Bouey(Task):
    def __init__(self, params: TaskParams) -> None:
        self.mu = params.mu
        self.cancelled = False
        self.status = params.status
        self.finder = Finder()

    def run(self, tag):
        if (self.cancelled): return
        pos = self.mu.get_position()
        bouey = self.finder.find_by_tag(tag)

        if(bouey==None):
            self.cancelled = True
            return

        bouey_pos = bouey.position

        p0 = Waypoint(pos)
        p1 = Waypoint(bouey_pos)
        p2 = Waypoint((bouey_pos.x-1, bouey_pos.y, bouey_pos.z))

        traj = LinearTrajectory([p0, p1, p2])

        self.status(f"Starting trajectory! ETA: {traj.get_duration():.1f}s")
        self.do_traj(traj)
        self.status(f"Bouey done :)")
            
    def do_traj(self, traj: LinearTrajectory):
        self.mu.set_trajectory(traj)
        while self.mu.get_motion_status() < TrajectoryStatus.FINISHED \
            and not self.cancelled:
            rospy.sleep(rospy.Duration(0.5))
            # TODO: print eta

    def cancel(self):
        self.mu.abort()
        self.cancelled = True
