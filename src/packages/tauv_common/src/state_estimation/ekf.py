from math import sin, cos, tan, pi
from enum import IntEnum
from typing import List, Optional

import rospy
import numpy as np


class StateIndex(IntEnum):
    X = 0
    Y = 1
    Z = 2
    VX = 3
    VY = 4
    VZ = 5
    AX = 6
    AY = 7
    AZ = 8
    YAW = 9
    PITCH = 10
    ROLL = 11
    VYAW = 12
    VPITCH = 13
    VROLL = 14


class EKF:
    NUM_FIELDS = 15

    def __init__(self, dvl_offset: np.array, imu_covariance: np.array, dvl_covariance: np.array):
        self._dvl_offset: np.array = dvl_offset
        self._imu_covariance: np.array = imu_covariance
        self._dvl_covariance: np.array = dvl_covariance

        self._state: np.array = np.zeros(EKF.NUM_FIELDS, float)
        self._covariance: np.array = 1e-9 * np.identity(EKF.NUM_FIELDS, float)

        self._last_measurement_time: Optional[rospy.Time] = None

    def get_state(self) -> np.array:
        return self._state.copy()

    def handle_imu_measurement(self, linear_acceleration: np.array, orientation: np.array, angular_velocity: np.array, timestamp: rospy.Time):
        if self._last_measurement_time is None:
            self._last_measurement_time = timestamp

        delta_t: float = timestamp.to_sec() - self._last_measurement_time.to_sec()

        self._last_measurement_time = timestamp

        self._extrapolate_state(delta_t)
        self._extrapolate_covariance(delta_t)

        H: np.array = self._get_H([#StateIndex.AX, StateIndex.AY, StateIndex.AZ,
                                   StateIndex.YAW, StateIndex.PITCH, StateIndex.ROLL,
                                   StateIndex.VYAW, StateIndex.VPITCH, StateIndex.VROLL])

        z: np.array = np.concatenate((orientation, angular_velocity))

        y: np.array = z - np.matmul(H, self._state)

        R: np.array = np.diag(self._imu_covariance)

        S: np.array = np.matmul(H, np.matmul(self._covariance, np.transpose(H))) + R

        K: np.array = np.matmul(self._covariance, np.matmul(np.transpose(H), np.linalg.inv(S)))

        I: np.array = np.identity(EKF.NUM_FIELDS, float)

        self._state = self._state + np.matmul(K, y)
        self._state[StateIndex.YAW] = (self._state[StateIndex.YAW] + pi) % (2 * pi) - pi
        self._state[StateIndex.PITCH] = (self._state[StateIndex.PITCH] + pi) % (2 * pi) - pi
        self._state[StateIndex.ROLL] = (self._state[StateIndex.ROLL] + pi) % (2 * pi) - pi

        self._covariance = np.matmul(I - np.matmul(K, H), self._covariance)
        self._covariance = np.maximum(np.abs(self._covariance), 1e-9 * np.identity(EKF.NUM_FIELDS, float))

    def handle_dvl_measurement(self, linear_velocity: np.array, timestamp: rospy.Time):
        if self._last_measurement_time is None:
            self._last_measurement_time = timestamp

        delta_t: float = timestamp.to_sec() - self._last_measurement_time.to_sec()

        self._last_measurement_time = timestamp

        self._extrapolate_state(delta_t)
        self._extrapolate_covariance(delta_t)

        H: np.array = self._get_H([StateIndex.VX, StateIndex.VY, StateIndex.VZ])

        z: np.array = linear_velocity - self._get_dvl_tangential_velocity()

        y: np.array = z - np.matmul(H, self._state)

        R: np.array = np.diag(self._dvl_covariance)

        S: np.array = np.matmul(H, np.matmul(self._covariance, np.transpose(H))) + R

        K: np.array = np.matmul(self._covariance, np.matmul(np.transpose(H), np.linalg.inv(S)))

        I: np.array = np.identity(EKF.NUM_FIELDS, float)

        self._state = self._state + np.matmul(K, y)
        self._state[StateIndex.YAW] = (self._state[StateIndex.YAW] + pi) % (2 * pi) - pi
        self._state[StateIndex.PITCH] = (self._state[StateIndex.PITCH] + pi) % (2 * pi) - pi
        self._state[StateIndex.ROLL] = (self._state[StateIndex.ROLL] + pi) % (2 * pi) - pi

        self._covariance = np.matmul(I - np.matmul(K, H), self._covariance)
        self._covariance = np.maximum(np.abs(self._covariance), 1e-9 * np.identity(EKF.NUM_FIELDS, float))

    def _extrapolate_state(self, dt: float):
        F: np.array = self._get_F(dt)

        self._state = np.matmul(F, self._state)
        self._state[StateIndex.YAW] = (self._state[StateIndex.YAW] + pi) % (2 * pi) - pi
        self._state[StateIndex.PITCH] = (self._state[StateIndex.PITCH] + pi) % (2 * pi) - pi
        self._state[StateIndex.ROLL] = (self._state[StateIndex.ROLL] + pi) % (2 * pi) - pi

    def _extrapolate_covariance(self, dt: float):
        J: np.array = self._get_J(dt)

        Q: np.array = 0.0001 * np.identity(EKF.NUM_FIELDS, float)

        self._covariance = np.matmul(J, np.matmul(self._covariance, np.transpose(J))) + Q

    def _get_dvl_tangential_velocity(self) -> np.array:
        cp: float = cos(self._state[StateIndex.PITCH])
        sp: float = sin(self._state[StateIndex.PITCH])
        cr: float = cos(self._state[StateIndex.ROLL])
        sr: float = sin(self._state[StateIndex.ROLL])

        vyaw: float = self._state[StateIndex.VYAW]
        vpitch: float = self._state[StateIndex.VPITCH]
        vroll: float = self._state[StateIndex.VROLL]

        w: np.array = np.array([-sp * vyaw + vroll,
                                cp * sr * vyaw + cr * vpitch,
                                cp * cr * vyaw - sr * vpitch])

        return np.cross(w, self._dvl_offset)

    def _get_H(self, fields: List[int]) -> np.array:
        H: np.array = np.zeros((len(fields), EKF.NUM_FIELDS), float)

        for i, f in enumerate(fields):
            H[i, f] = 1

        return H

    def _get_F(self, dt: float) -> np.array:
        cy: float = cos(self._state[StateIndex.YAW])
        sy: float = sin(self._state[StateIndex.YAW])
        cp: float = cos(self._state[StateIndex.PITCH])
        sp: float = sin(self._state[StateIndex.PITCH])
        tp: float = tan(self._state[StateIndex.PITCH])
        cr: float = cos(self._state[StateIndex.ROLL])
        sr: float = sin(self._state[StateIndex.ROLL])

        F: np.array = np.identity(EKF.NUM_FIELDS, float)
        F[StateIndex.X, StateIndex.VX] = (cp * cy) * dt
        F[StateIndex.X, StateIndex.VY] = (cy * sp * sr - cr * sy) * dt
        F[StateIndex.X, StateIndex.VZ] = (cr * cy * sp + sr * sy) * dt
        F[StateIndex.X, StateIndex.AX] = 0.5 * F[StateIndex.X, StateIndex.VX] * dt
        F[StateIndex.X, StateIndex.AY] = 0.5 * F[StateIndex.X, StateIndex.VY] * dt
        F[StateIndex.X, StateIndex.AZ] = 0.5 * F[StateIndex.X, StateIndex.VZ] * dt
        F[StateIndex.Y, StateIndex.VX] = (cp * sy) * dt
        F[StateIndex.Y, StateIndex.VY] = (cp * cy + sp * sr * sy) * dt
        F[StateIndex.Y, StateIndex.VZ] = (-cy * sr + cr * sp * sy) * dt
        F[StateIndex.Y, StateIndex.AX] = 0.5 * F[StateIndex.Y, StateIndex.VX] * dt
        F[StateIndex.Y, StateIndex.AY] = 0.5 * F[StateIndex.Y, StateIndex.VY] * dt
        F[StateIndex.Y, StateIndex.AZ] = 0.5 * F[StateIndex.Y, StateIndex.VZ] * dt
        F[StateIndex.Z, StateIndex.VX] = (-sp) * dt
        F[StateIndex.Z, StateIndex.VY] = (cp * sr) * dt
        F[StateIndex.Z, StateIndex.VZ] = (cp * cr) * dt
        F[StateIndex.Z, StateIndex.AX] = 0.5 * F[StateIndex.Z, StateIndex.VX] * dt
        F[StateIndex.Z, StateIndex.AY] = 0.5 * F[StateIndex.Z, StateIndex.VY] * dt
        F[StateIndex.Z, StateIndex.AZ] = 0.5 * F[StateIndex.Z, StateIndex.VZ] * dt
        F[StateIndex.VX, StateIndex.AX] = dt
        F[StateIndex.VY, StateIndex.AY] = dt
        F[StateIndex.VZ, StateIndex.AZ] = dt
        F[StateIndex.YAW, StateIndex.VYAW] = (cr / cp) * dt
        F[StateIndex.YAW, StateIndex.VPITCH] = (sr / cp) * dt
        F[StateIndex.PITCH, StateIndex.VYAW] = (-sr) * dt
        F[StateIndex.PITCH, StateIndex.VPITCH] = (cr) * dt
        F[StateIndex.ROLL, StateIndex.VYAW] = (cr * tp) * dt
        F[StateIndex.ROLL, StateIndex.VPITCH] = (sr * tp) * dt
        F[StateIndex.ROLL, StateIndex.VROLL] = dt

        return F

    def _get_J(self, dt: float) -> np.array:
        J: np.array = self._get_F(dt)

        cy: float = cos(self._state[StateIndex.YAW])
        sy: float = sin(self._state[StateIndex.YAW])
        cp: float = cos(self._state[StateIndex.PITCH])
        sp: float = sin(self._state[StateIndex.PITCH])
        tp: float = tan(self._state[StateIndex.PITCH])
        cr: float = cos(self._state[StateIndex.ROLL])
        sr: float = sin(self._state[StateIndex.ROLL])

        J[StateIndex.X, StateIndex.YAW] = self._get_partial(-cp * sy,
                                                            -cy * cr - sp * sy * sr,
                                                            cy * sr - sp * sy * cr,
                                                            dt)
        J[StateIndex.X, StateIndex.PITCH] = self._get_partial(-sp * cy,
                                                              cp * cy * sr,
                                                              cp * cy * cr,
                                                              dt)
        J[StateIndex.X, StateIndex.ROLL] = self._get_partial(0,
                                                             sp * cy * cr + sy * sr,
                                                             sy * cr - sp * cy * sr,
                                                             dt)
        J[StateIndex.Y, StateIndex.YAW] = self._get_partial(cp * cy,
                                                            sp * cy * sr - sy * cr,
                                                            sp * cy * sr + sy * sr,
                                                            dt)
        J[StateIndex.Y, StateIndex.PITCH] = self._get_partial(-sp * sy,
                                                              cp * sy * sr,
                                                              cp * sy * cr,
                                                              dt)
        J[StateIndex.Y, StateIndex.ROLL] = self._get_partial(0,
                                                             sp * sy * cr - cy * sr,
                                                             -cy * cr - sp * sy * sr,
                                                             dt)
        J[StateIndex.Z, StateIndex.PITCH] = self._get_partial(-cp,
                                                              -sp * sr,
                                                              -sp * cr,
                                                              dt)
        J[StateIndex.Z, StateIndex.ROLL] = self._get_partial(0,
                                                             cp * sr,
                                                             -cp * sr,
                                                             dt)

        vyaw: float = self._state[StateIndex.VYAW]
        vpitch: float = self._state[StateIndex.VPITCH]

        J[StateIndex.YAW, StateIndex.PITCH] = (tp * sr / cp) * vpitch * dt + (tp * cr / cp) * vyaw * dt
        J[StateIndex.YAW, StateIndex.ROLL] = (cr / cp) * vpitch * dt + (-sr / cp) * vyaw * dt
        J[StateIndex.PITCH, StateIndex.ROLL] = (-sr) * vpitch * dt + (-cr) * vyaw * dt
        J[StateIndex.ROLL, StateIndex.PITCH] = (sr / (cp * cp)) * vpitch * dt + (cr / (cp * cp)) * vyaw * dt
        J[StateIndex.ROLL, StateIndex.ROLL] = 1 + (tp * cr) * vpitch * dt + (-tp * sr) * vyaw * dt

        return J

    def _get_partial(self, xc: float, yc: float, zc: float, dt: float) -> float:
        vx: float = self._state[StateIndex.VX]
        vy: float = self._state[StateIndex.VY]
        vz: float = self._state[StateIndex.VZ]
        ax: float = self._state[StateIndex.AX]
        ay: float = self._state[StateIndex.AY]
        az: float = self._state[StateIndex.AZ]

        return (xc * vx + yc * vy + zc * vz) * dt + \
               (xc * ax + yc * ay + zc * az) * 0.5 * dt * dt