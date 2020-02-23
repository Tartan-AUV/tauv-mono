# opt_planner.py
#
# An optimal planner for trajectory optimization.
#
#
#!/usr/bin/env python

import rospy
import cvxpy as cvx
import numpy as np
from tauv_common.planner.opt_planner.opt_planner_utils import State3d, Control3d
from geometry_msgs.msg import Accel
from nav_msgs.msg import Odometry
import tf.transformations as angle_transform


N = State3d.dim
M = Control3d.dim

# Horizon and timesteps
horizon = 4     # seconds
Dt = 0.2        # seconds
# number of timesteps
T = int(horizon / Dt)

One = 1.0
Xd = (1.0 / 2.0) * Dt**2

# Linear System Data
A = np.array([[One, 0.0, 0.0, 0.0, 0.0, 0.0, Dt,  0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, One, 0.0, 0.0, 0.0, 0.0, 0.0,  Dt, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, One, 0.0, 0.0, 0.0, 0.0, 0.0,  Dt, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, One, 0.0, 0.0, 0.0, 0.0, 0.0,  Dt, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, One, 0.0, 0.0, 0.0, 0.0, 0.0,  Dt, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, One, 0.0, 0.0, 0.0, 0.0, 0.0,  Dt],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, One, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, One, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, One, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, One, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, One, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, One]])


B = np.array([[ Xd, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0,  Xd, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0,  Xd, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0,  Xd, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0,  Xd, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0,  Xd],
              [ Dt, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0,  Dt, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0,  Dt, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0,  Dt, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0,  Dt, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0,  Dt]])


class OptPlanner():
    def __init__(self):
        self.pub_accel = rospy.Publisher("cmd_acc", Accel, queue_size=1)
        self.sub_odom = rospy.Subscriber("/gnc/odom", Odometry, self.odom_callback)

        # Initial condition
        self.x_0 = np.zeros(N)

        self.states = np.zeros((T + 1, State3d.dim))
        self.controls = np.zeros((T, Control3d.dim))

    def create_constrained_problem(self, init_state, final_state):
        """
        Form planning problem.
        """
        assert init_state.shape == (State3d.dim,)
        assert final_state.shape == (State3d.dim,)

        x = cvx.Variable((N, T + 1))
        u = cvx.Variable((M, T))

        cost = 0.0
        constr = []

        # INITIAL CONDITION CONSTRAINTS
        constr += [self.x[:, 0] == init_state]

        # INTERIM COST AND CONSTRAINTS
        for t in range(T):
            # STAGE COSTS
            cost += cvx.sum_squares(self.u[:,t])

            # DYNAMICS
            constr += [self.x[:, t + 1] == A * self.x[:, t] + B * self.u[:, t]]

        # FINAL STATE COST and CONSTRAINTS

        # Final State Constraints
        constr += [self.x[:, T] == final_state]
        planning_problem = cvx.Problem(cvx.Minimize(cost), constr)
        return planning_problem, x, u

    def solve_planning_problem(self, planning_problem, x, u, debug=False):
        # PROBLEM DEFINITION
        planning_problem.solve(solver=cvx.ECOS, verbose=debug)
        if debug:
            print ("Solver Time  : ", planning_problem.solver_stats.solve_time)
            print ("Optimal Cost : ", planning_problem.value)

        states = np.transpose(x.value)
        controls = np.transpose(u.value)
        return states, controls

    def odom_callback(self, msg):
        ### Position
        pose = msg.pose.pose
        ## linear
        self.x0[State3d.x] = pose.position.x
        self.x0[State3d.y] = pose.position.y
        self.x0[State3d.z] = pose.position.z
        ## angular
        quaternion = (pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w)

        euler = angle_transform.euler_from_quaternion(quaternion)
        self.x0[State3d.roll] = euler[0]
        self.x0[State3d.pitch] = euler[1]
        self.x0[State3d.yaw] = euler[2]

        ### Velocity
        twist = msg.twist.twist
        ## linear
        self.x0[State3d.x_dot] = twist.linear.x
        self.x0[State3d.y_dot] = twist.linear.y
        self.x0[State3d.z_dot] = twist.linear.z
        ## angular
        self.x0[State3d.roll_dot] = twist.angular.x
        self.x0[State3d.pitch_dot] = twist.angular.y
        self.x0[State3d.yaw_dot] = twist.angular.z

    def plan_trajectory(self, final_state):
        prob, x, u = self.create_constrained_problem(self.x0, final_state)
        states, controls = self.solve_planning_problem(prob, x, u)
        accel_msg = self.cvx_accel_to_accel_msg(controls)
        self.pub_accel.publish(accel_msg)
        return states, controls

    def cvx_accel_to_accel_msg(self, u):
        msg = Accel()
        msg.linear.x = u[0, Control3d.a_x]
        msg.linear.y = u[0, Control3d.a_y]
        msg.linear.z = u[0, Control3d.a_z]

        msg.angular.x = u[0, Control3d.a_roll]
        msg.angular.y = u[0, Control3d.a_pitch]
        msg.angular.z = u[0, Control3d.a_yaw]
        return msg
