import numpy as np
from typing import Callable
from math import sqrt


class SCurve:
    def __init__(self, q1: float, v_max: float, a_max: float, j_max: float):
        self._ak: float = 0.0
        self._q1: float = q1
        self._v_max: float = v_max
        self._a_max: float = a_max
        self._j_max: float = j_max

    def plan(self, q0: float, v0: float) -> (Callable[[float], float], Callable[[float], float], Callable[[float], float], Callable[[float], float]):
        t_i_decel = 0.0
        t_i_decel_jerk = 0.0

        if v0 > self._v_max:
            # Need initial decel
            t_i_decel_jerk = (self._a_max / self._j_max)
            t_i_decel = t_i_decel_jerk + (v0 - self._v_max) / self._a_max

        a_lim_i_decel = -self._j_max * t_i_decel_jerk

        t_decel = 0.0
        t_decel_jerk = 0.0
        # 3.20
        if self._v_max * self._j_max < self._a_max ** 2:
            # a_min not reached on decel
            # 3.23
            t_decel_jerk = sqrt(self._v_max / self._j_max)
            t_decel = 2 * t_decel_jerk
        else:
            # a_min reached on decel
            # 3.24
            t_decel_jerk = self._a_max / self._j_max
            t_decel = t_decel_jerk + (self._v_max / self._a_max)

        a_lim_decel = -self._j_max * t_decel_jerk

        # def v(t: float):
        #     if t < t_i_decel_jerk:
        #         return v0 - self._j_max * ((t ** 2) / 2)
        #     elif t < t_i_decel - t_i_decel_jerk:
        #         return v0 + a_lim_i_decel * (t - t_i_decel_jerk / 2)
        #     elif t < t_i_decel:
        #         return self._v_max + self._j_max * ((t_i_decel - t) ** 2) / 2
        #
        #     t = t - t_i_decel
        #     if t < t_decel_jerk:
        #         return self._v_max - self._j_max * (t ** 2) / 2
        #     elif t < t_decel - t_decel_jerk:
        #         return self._v_max + a_lim_decel * (t - t_decel_jerk / 2)
        #     elif t < t_decel:
        #         return self._j_max * ((t_decel - t) ** 2) / 2
        #
        #     return 0.0

        def a(t: float):
            if t < t_i_decel_jerk:
                return -self._j_max * t
            elif t < t_i_decel - t_i_decel_jerk:
                return a_lim_i_decel
            elif t < t_i_decel:
                return -self._j_max * (t_i_decel - t)

            t = t - t_i_decel
            if t < t_decel_jerk:
                return -self._j_max * t
            elif t < t_decel - t_decel_jerk:
                return a_lim_decel
            elif t < t_decel:
                return -self._j_max * (t_decel - t)

            return 0.0

        def v(t: float):
            if t <= 0.0:
                return v0
            else:
                return 0.05 * a(t) + v(t - 0.05)

        def q(t: float):
            if t <= 0.0:
                return q0
            else:
                return 0.05 * v(t) + q(t - 0.05)

        def j(t: float):
            if t < t_i_decel_jerk:
                return -self._j_max
            elif t < t_i_decel - t_i_decel_jerk:
                return 0.0
            elif t < t_i_decel:
                return self._j_max

            t = t - t_i_decel
            if t < t_decel_jerk:
                return -self._j_max
            elif t < t_decel - t_decel_jerk:
                return 0.0
            elif t < t_decel:
                return self._j_max

            return 0.0

        return (q, v, a, j)
