import rospy
import numpy as np
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import dynamics_utils


from tauv_msgs.msg import NavigationState
from geometry_msgs.msg import Wrench

class DynamicsTuner:

    def __init__(self):
        self._navigation_state_sub: rospy.Subscriber = rospy.Subscriber('/gnc/state_estimation/navigation_state', NavigationState, self._handle_navigation_state)
        self._wrench_sub: rospy.Subscriber = rospy.Subscriber('/vehicle/thrusters/wrench', Wrench, self._handle_wrench)

        self._n_nav_states = 1000
        self._n_wrenches = 1000
        self._i_nav_states = 0
        self._i_wrenches = 0

        self._nav_states: np.array = np.zeros((self._n_nav_states, 7))
        self._wrenches: np.array = np.zeros((self._n_wrenches, 7))
        self._processing: bool = False

        # Accumulate

    def start(self):
        rospy.spin()

    def _handle_navigation_state(self, msg: NavigationState):
        if self._processing:
            return

        self._nav_states[self._i_nav_states, :] = np.array([
            msg.header.stamp.to_sec(),
            msg.velocity.x,
            msg.velocity.y,
            msg.velocity.z,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ])

        if self._i_nav_states < self._n_nav_states - 1:
            self._i_nav_states += 1
        else:
            self._process()
            # process


    def _handle_wrench(self, msg: Wrench):
        if self._processing:
            return

        self._wrenches[self._i_wrenches, :] = np.array([
            rospy.Time.now().to_sec(),
            msg.force.x,
            msg.force.y,
            msg.force.z,
            msg.torque.x,
            msg.torque.y,
            msg.torque.z
        ])

        if self._i_wrenches < self._n_wrenches - 1:
           self._i_wrenches += 1
        else:
            self._process()

    def _process(self):
        self._processing = True

        # Interpolate wrenches to match nav states
        nav_state_times = self._nav_states[0:self._i_nav_states, 0]
        wrench_times = self._wrenches[:, 0]
        wrenches = self._wrenches[:, 1:7]

        wrenches_f = interpolate.interp1d(wrench_times, wrenches, axis=0, fill_value=0.0, bounds_error=False)
        wrenches_interp = wrenches_f(nav_state_times)

        valid_nav_states = self._nav_states[0:self._i_nav_states, :]
        linear_velocity = valid_nav_states[:, 1:4]
        orientation = valid_nav_states[:, 4:7]

        linear_acceleration = np.vstack([np.diff(linear_velocity, axis=0), np.zeros(3)])
        angular_velocity = np.vstack([np.diff(orientation, axis=0), np.zeros(3)])
        angular_acceleration = np.vstack([np.diff(angular_velocity, axis=0), np.zeros(3)])

        force = wrenches_interp[:, 0:3]
        torque = wrenches_interp[:, 3:6]

        # define error as a function of the dynamics parameters
        # sum-squared of each individual error term?

        u = np.hstack([orientation, linear_velocity, angular_velocity, force, torque])
        y = np.hstack([linear_acceleration, angular_acceleration])

        # v, r_G, r_B, M_I, L
        def loss(parameters: np.array) -> float:
            m = 30

            '''
            rpy = u(1, 1:3);
            lin_v = u(1, 4:6);
            ang_v = u(1, 7:9);
            tau = u(1, 10:15);

            I_t = buildInertiaTensor(I(1), I(2), I(3), 0, 0, 0);
            M_rb = buildMassMatrix(m, CG, I_t);
            C_rb = buildCoriolisMatrix(m, CG, I_t, lin_v.
            ', ang_v.');
            D_lin = diag([L]);
            G = buildGravityMatrix(m, v, CG, CB, rpy.
            ');

            y = (M_rb \ (tau.' - G - C_rb * [lin_v.'; ang_v.'] - D_lin * [lin_v.'; ang_v.'])).';
            '''

            orientation = np.flip(u[:, 0:3], axis=1)
            v = np.hstack([
                u[:, 3:6],
                np.flip(u[:, 6:9], axis=1)
            ])
            tau = np.hstack([
                u[:, 9:12],
                np.flip(u[:, 12:15], axis=1)
            ])

            vol = parameters[0]
            r_G = parameters[1:4]
            r_B = parameters[4:7]
            M_I = parameters[7:13]
            L = parameters[13:19]

            err = np.zeros(u.shape[0])

            for i in range(u.shape[0]):
                I = dynamics_utils.I(M_I)
                M_rb = dynamics_utils.M_rb(m, r_G, I)
                C_rb = dynamics_utils.C_rb(m, v[i], r_G, I)
                D = np.diag(L)
                G = dynamics_utils.G(m, vol, r_G, r_B, np.flip(orientation[i]))

                y_est = np.linalg.inv(M_rb) @ (np.transpose(tau[i]) - np.transpose(G) - C_rb @ v[i] - D @ v[i])

                y_err = (y_est - y[i])
                sum_square_err = np.sum(y_err ** 2, axis=0)
                err[i] = sum_square_err

            total_err = np.sum(err ** 2) / err.shape[0]
            return total_err

        vol_init = np.array([30])
        r_G_init = np.array([0, 0, 0])
        r_B_init = np.array([0, 0, 0])
        M_I_init = np.array([10, 10, 10, 0, 0, 0])
        L_init = np.array([10, 10, 10, 10, 10, 10])

        initial_parameters = np.concatenate((vol_init, r_G_init, r_B_init, M_I_init, L_init))
        opt_result = optimize.minimize(loss, initial_parameters, method='trust-constr', options={'maxiter': 50, 'vebose': 3})

        print(opt_result)

        vol_est = opt_result.x[0]
        r_G_est = opt_result.x[1:4]
        r_B_est = opt_result.x[4:7]
        M_I_est = opt_result.x[7:13]
        L_est = opt_result.x[13:19]

        print('vol:', vol_est)
        print('r_G:', r_G_est)
        print('r_B:', r_B_est)
        print('M_I:', M_I_est)
        print('L:', L_est)

        # Compute linear accel
        # Compute angular velocity
        # Compute angular accel
        # Filter everything

        # TODO: do optimization

        self._nav_states: np.array = np.zeros((self._n_nav_states, 7))
        self._wrenches: np.array = np.zeros((self._n_wrenches, 7))
        self._i_nav_states = 0
        self._i_wrenches = 0
        self._processing = False


def main():
    rospy.init_node('dynamics_tuner')
    d = DynamicsTuner()
    d.start()

if __name__ == "__main__":
    main()
