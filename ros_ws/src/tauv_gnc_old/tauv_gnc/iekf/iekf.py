import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm

# Adapted from https://bitbucket.org/frostlab/underwateriekf/src/main/src/iekf.py

class IEKF:
    def __init__(self, system, mu_0, sigma_0, left=False):
        """The newfangled Invariant Extended Kalman Filter

        Args:
            system      (class) : The system to run the iEKF on. It will pull Q, R, f, h from this.
            mu0     (nxn array) : Initial starting point of system
            sigma0  (mxm array) : Initial covariance of system"""
        if mu_0.shape == (9,):
            mu_0 = expm(system.carat(mu_0))

        self.sys = system
        self.mus = [mu_0]
        self.sigmas = [sigma_0]
        self.biass = [np.zeros((2, 3))]

        self.invR = np.zeros((3, 3))
        self.invR[-1, -1] = 1 / self.sys.R[3, 3]

        self.left = left

    # These 3 properties are helpers. Since our mus and sigmas are stored in a list
    # it can be a pain to access the last one. These provide "syntactic sugar"
    @property
    def mu(self):
        return self.mus[-1]

    @property
    def sigma(self):
        return self.sigmas[-1]

    @property
    def bias(self):
        return self.biass[-1]

    def predict(self, u, dt):
        """Runs prediction step of iEKF.

        Args:
            u       (k ndarray) : control taken at this step

        Returns:
            mu    (nxn ndarray) : Propagated state
            sigma (nxn ndarray) : Propagated covariances"""

        # get mubar and sigmabar
        mu_bar = self.sys.f_lie(self.mu, u, dt, self.bias)

        # make adjoint
        I = np.eye(6)
        zero = np.zeros((9, 6))
        adj_X = np.block([[self.sys.adjoint(mu_bar), zero],
                          [zero.T, I]])

        # make propagation matrix
        zero = np.zeros((3, 3))
        I = np.eye(3)
        if self.left:
            w_cross = self.sys.cross(u[0] - self.bias[0])
            a_cross = self.sys.cross(u[1] - self.bias[1])
            self.expA = expm(np.block([[-w_cross, zero, zero, -I, zero],
                                       [-a_cross, -w_cross, zero, zero, -I],
                                       [zero, I, -w_cross, zero, zero],
                                       [zero, zero, zero, zero, zero],
                                       [zero, zero, zero, zero, zero]]) * dt)
            sigma_bar = self.expA @ self.sigma @ self.expA.T + self.expA @ self.sys.Q @ self.expA.T * dt

        else:
            R = mu_bar[:3, :3]
            g_cross = self.sys.cross(np.array([0, 0, -9.8]))
            # v_cross = self.sys.cross( mu_bar[:3,3] )
            # p_cross = self.sys.cross( mu_bar[:3,4] )
            self.expA = expm(np.block([[zero, zero, zero, -R, zero],
                                       [g_cross, zero, zero, -adj_X[3:6, 0:3], -R],
                                       [zero, I, zero, -adj_X[6:9, 0:3], zero],
                                       [zero, zero, zero, zero, zero],
                                       [zero, zero, zero, zero, zero]]) * dt)
            sigma_bar = self.expA @ (self.sigma + adj_X @ self.sys.Q @ adj_X.T * dt) @ self.expA.T

        # save for use later
        self.last_u = u
        self.mus.append(mu_bar)
        self.biass.append(self.bias)
        self.sigmas.append(sigma_bar)

        return mu_bar, sigma_bar

    def update_depth(self, z, sim_bias=True):
        """Runs correction step of iEKF.

        Args:
            z (m ndarray): measurement at this step

        Returns:
            mu    (nxn ndarray) : Corrected state
            sigma (nxn ndarray) : Corrected covariances"""
        # make H matrices
        zero = np.zeros((3, 3))
        I = np.eye(3)
        H = np.block([zero, zero, I, zero, zero])

        I = np.eye(6)
        zero = np.zeros((9, 6))
        # convert H to Right measurement
        if not self.left:
            H = H @ np.block([[self.sys.adjoint(inv(self.mu)), zero],
                              [zero.T, I]])

        # put our measurements into full form
        z = np.array([self.mu[0, 4], self.mu[1, 4], z, 0, 1])

        # make innovation
        V = (inv(self.mu) @ z)[:3]

        R = self.mu[:3, :3]

        # make our special measurement covariance
        sig_til = inv(H @ self.sigma @ H.T)
        meas_cov = sig_til - sig_til @ inv(R.T @ self.invR @ R + sig_til) @ sig_til

        K = self.sigma @ H.T @ meas_cov
        K_state = K[:9]
        K_bias = K[9:]
        if self.left:
            self.mus[-1] = self.mu @ expm(self.sys.carat(K @ V))
        else:
            self.mus[-1] = expm(self.sys.carat(K @ V)) @ self.mu
        if sim_bias:
            self.biass[-1] = self.bias + (K_bias @ V).reshape((2, 3))
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma

        return self.mu, self.sigma

    def update_dvl(self, z, sim_bias=True):
        """Runs correction step of iEKF.

        Args:
            z (m ndarray): measurement at this step

        Returns:
            mu    (nxn ndarray) : Corrected state
            sigma (nxn ndarray) : Corrected covariances"""
        # convert dvl into correct frame
        z = self.sys.dvl_r @ z + self.sys.dvl_p @ (self.last_u[0] - self.bias[0])

        # make H matrices
        zero = np.zeros((3, 3))
        I = np.eye(3)
        H = np.block([zero, I, zero, zero, zero])

        I = np.eye(6)
        zero = np.zeros((9, 6))
        # convert H1 to Left measurement
        if self.left:
            H = H @ np.block([[self.sys.adjoint(self.mu), zero],
                              [zero.T, I]])

        # put our measurements into full form
        z = np.array([z[0], z[1], z[2], -1, 0])

        # make innovation
        V = (self.mu @ z)[:3]

        R = self.mu[:3, :3]

        # make our special measurement covariance
        meas_cov = inv(H @ self.sigma @ H.T + R @ self.sys.R[:3, :3] @ R.T)

        K = self.sigma @ H.T @ meas_cov
        K_state = K[:9]
        K_bias = K[9:]
        if self.left:
            self.mus[-1] = self.mu @ expm(self.sys.carat(K_state @ V))
        else:
            self.mus[-1] = expm(self.sys.carat(K_state @ V)) @ self.mu
        if sim_bias:
            self.biass[-1] = self.bias + (K_bias @ V).reshape((2, 3))
        self.sigmas[-1] = (np.eye(15) - K @ H) @ self.sigma

        return self.mu, self.sigma
