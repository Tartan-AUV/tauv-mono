import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv

# Adapted from https://bitbucket.org/frostlab/underwateriekf/src/main/src/system.py

class IEKFSystemKinematic:
    def __init__(self, Q, R, dvl_p=np.zeros(3), dvl_r=np.eye(3)):
        """Our system for a quadcopter, with IMU measurements as controls.

        Args:
            filename      (str) : Where the recorded data has been stored
            Q   (15,15 ndarray) : Covariance of noise on state
            R     (6,6 ndarray) : Covariance of noise on measurements
            dvl_p   (3 ndarray) : Translation of DVL from IMU
            dvl_r (3,3 ndarray) : Rotation matrix of DVL from IMU"""
        self.Q = Q
        self.R = R
        self.Hz = self.data['ticks']
        self.T = self.data['x'].shape[0]
        # local velocity
        self.b1 = np.array([0, 0, 0, -1, 0])
        # global depth
        self.b2 = np.array([0, 0, 0, 0, 1])

        self.dvl_p = IEKFSystemKinematic.cross(dvl_p)
        self.dvl_r = dvl_r

        # convert z1 noise into the correct frame
        self.R[:3, :3] = self.dvl_r @ self.R[:3, :3] @ self.dvl_r.T + self.dvl_p @ (
                    self.Q[0:3, 0:3] + self.Q[9:12, 9:12]) @ self.dvl_p.T


    def f_lie(self, state, u, dt, bias):
        """Propagates state forward in Lie Group. Used for IEKF.

        Args:
            state (5,5 ndarray) : X_n of model in Lie Group
            u     (2,3 ndarray) : U_n of model (IMU measurements)
            bias  (2,3 ndarray) : Estimated bias

        Returns:
            X_{n+1} (5,5 ndarray)"""
        # transform u into the right representation
        # get stuff we need
        g = np.array([0, 0, -9.8])
        R = state[:3, :3]
        v = state[:3, 3]
        p = state[:3, 4]

        omega = u[0] - bias[0]
        a = u[1] - bias[1]

        ## put it together
        Rnew = R @ expm(self.cross(omega * dt))
        vnew = v + (R @ a + g) * dt
        pnew = p + v * dt + (R @ a + g) * dt ** 2 / 2

        return np.block([[Rnew, vnew.reshape(-1, 1), pnew.reshape(-1, 1)],
                         [np.zeros((2, 3)), np.eye(2)]])

    def h(self, state):
        """Calculates measurement given a state. Note that the result is
            the same if it's in standard or Lie Group form, so we simplify into
            one function.

        Args:
            state (3,3 ndarray) : Current state in either standard or Lie Group form
            noise        (bool) : Whether or not to add noise. Defaults to False.

        Returns:
            z1 (3 ndarray) : DVL Velocity measurement
            z2   (1 float) : Depth measurement"""
        # local velocity
        z1 = (inv(state) @ self.b1)[:3]
        # global depth
        z2 = ((state) @ self.b2)[2]

        return z1, z2

    @staticmethod
    def cross(x):
        """Moves a 3 vector into so(3)

        Args:
            x (3 ndarray) : Parametrization of Lie Algebra

        Returns:
            x (3,3 ndarray) : Element of so(3)"""
        return np.array([[0, -x[2], x[1]],
                         [x[2], 0, -x[0]],
                         [-x[1], x[0], 0]])

    @staticmethod
    def carat(xi):
        """Moves a 9 vector to the Lie Algebra se_2(3).

        Args:
            xi (9 ndarray) : Parametrization of Lie algebra

        Returns:
            xi^ (5,5 ndarray) : Element in Lie Algebra se_2(3)"""
        w_cross = IEKFSystemKinematic.cross(xi[0:3])
        v = xi[3:6].reshape(-1, 1)
        p = xi[6:9].reshape(-1, 1)
        return np.block([[w_cross, v, p],
                         [np.zeros((2, 5))]])

    @staticmethod
    def adjoint(xi):
        """Takes adjoint of element in SE_2(3)

        Args:
            xi (5,5 ndarray) : Element in Lie Group

        Returns:
            Ad_xi (9,9 ndarray) : Adjoint in SE_2(3)"""
        R = xi[:3, :3]
        v_cross = IEKFSystemKinematic.cross(xi[:3, 3])
        p_cross = IEKFSystemKinematic.cross(xi[:3, 4])
        zero = np.zeros((3, 3))
        return np.block([[R, zero, zero],
                         [v_cross @ R, R, zero],
                         [p_cross @ R, zero, R]])