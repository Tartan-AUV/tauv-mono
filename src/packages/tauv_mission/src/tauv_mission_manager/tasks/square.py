from motion.trajectories.trajectories import TrajectoryStatus
import rospy
from tauv_mission_manager.mission_utils import Task, TaskParams
from motion.trajectories.linear_trajectory import LinearTrajectory, Waypoint

class Square(Task):
    def __init__(self, params: TaskParams) -> None:
        self.mu = params.mu
        self.cancelled = False
        self.status = params.status

    def run(self, width):
        if (self.cancelled): return
        pos = self.mu.get_position()
        p0 = Waypoint(pos)
        p1 = Waypoint((pos[0]+width, pos[1], pos[2]))
        p2 = Waypoint((pos[0]+width, pos[1]+width, pos[2]))
        p3 = Waypoint((pos[0], pos[1]+width, pos[2]))
        p4 = Waypoint(pos)

        traj = LinearTrajectory([p0, p1, p2, p3, p4])

        self.status(f"Starting trajectory! ETA: {traj.get_duration():.1f}s")
        self.do_traj(traj)
        self.status(f"Square done :)")
            
    def do_traj(self, traj: LinearTrajectory):
        self.mu.set_trajectory(traj)
        while self.mu.get_motion_status() < TrajectoryStatus.FINISHED \
            and not self.cancelled:
            rospy.sleep(rospy.Duration(0.5))
            # TODO: print eta

    def cancel(self):
        self.mu.abort()
        self.cancelled = True
