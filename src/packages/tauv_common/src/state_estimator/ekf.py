from math import sin, cos, tan, pi
from enum import IntEnum
from typing import Optional

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

N_FIELDS = 15

class EKF:

    def __init__(self, process_covariance: np.array):
        self._process_covariance: np.array = process_covariance

        self._time: Optional[float] = None

        self._state: np.array = np.zeros(N_FIELDS, np.float32)
        self._covariance: np.array = 1e-9 * np.identity(N_FIELDS, np.float32)

    def get_state(self, time: float) -> Optional[np.array]:
        if self._time is None or time <= self._time:
            return None

        dt = time - self._time

        return self._extrapolate_state(dt)

    def handle_measurement(self, time: float, fields: [int], measurement: np.array, covariance: np.array):
        self._predict(time)

        H = self._get_H(fields)
        y = measurement - H @ self._state
        self._update(y, covariance, fields)

    def _predict(self, time: float):
        if self._time is None:
            self._time = time

        if time < self._time:
            raise ValueError("time < self._time")

        dt = time - self._time
        self._time = time
        print(dt)

        self._state = self._extrapolate_state(dt)
        self._covariance = self._extrapolate_covariance(dt)

    def _update(self, innovation: np.array, covariance: np.array, fields: [int]):
        H = self._get_H(fields)
        y = innovation
        R = np.diag(covariance)
        S = (H @ self._covariance) @ np.transpose(H) + R
        K = (self._covariance @ np.transpose(H)) @ np.linalg.inv(S)
        I = np.identity(N_FIELDS, np.float32)

        self._state = self._state + K @ y
        self._wrap_angles(self._state, [StateIndex.YAW, StateIndex.PITCH, StateIndex.ROLL])

        self._covariance = (I - (K @ H)) @ self._covariance
        # self._covariance = ((I - (K @ H)) @ self._covariance) @ np.transpose(I - (K @ H))
        # self._covariance = self._covariance + (K @ R) @ np.transpose(K)
        self._covariance = np.maximum(np.abs(self._covariance), 1e-9 * np.identity(N_FIELDS, np.float32))

    def _extrapolate_state(self, dt: float) -> np.array:
        F = self._get_F(dt)
        extrapolated_state = F @ self._state
        self._wrap_angles(extrapolated_state, [StateIndex.YAW, StateIndex.PITCH, StateIndex.ROLL])
        return extrapolated_state

    def _extrapolate_covariance(self, dt: float) -> np.array:
        J = self._get_J(dt)
        # extrapolated_covariance = J @ (self._covariance @ np.transpose(J)) + dt * self._process_covariance
        extrapolated_covariance = J @ (self._covariance @ np.transpose(J)) + self._process_covariance
        return extrapolated_covariance

    def _wrap_angles(self, state: np.array, fields: [int]) -> None:
        state[fields] = (state[fields] + pi) % (2 * pi) - pi

    def _get_H(self, fields: [int]) -> np.array:
        H: np.array = np.zeros((len(fields), N_FIELDS), np.float32)

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

        F: np.array = np.identity(N_FIELDS, float)
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
