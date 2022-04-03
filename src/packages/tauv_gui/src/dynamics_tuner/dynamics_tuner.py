import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dynamics.dynamics import Dynamics
from matplotlib.widgets import TextBox


class DynamicsTuner:

    def __init__(self):
        self._curr_axis = 0
        self._axes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        self._vel_fields = [
            '/gnc/pose/velocity/x',
            '/gnc/pose/velocity/y',
            '/gnc/pose/velocity/z',
            '/gnc/pose/angular_velocity/x',
            '/gnc/pose/angular_velocity/y',
            '/gnc/pose/angular_velocity/z',
        ]
        self._acc_fields = [
            '/gnc/pose/acceleration/x',
            '/gnc/pose/acceleration/y',
            '/gnc/pose/acceleration/z',
            '/gnc/pose/angular_acceleration/x',
            '/gnc/pose/angular_acceleration/y',
            '/gnc/pose/angular_acceleration/z',
        ]
        self._thrust_fields = [
            '/thrusters/wrench/force/x',
            '/thrusters/wrench/force/y',
            '/thrusters/wrench/force/z',
            '/thrusters/wrench/torque/x',
            '/thrusters/wrench/torque/y',
            '/thrusters/wrench/torque/z',
        ]

        self._data_dir = 'data'

        self.m = 21.2
        self.v = 0.0214
        self.rho = 1028.0
        self.r_G = np.array([0.0, 0.0, 0.0])
        self.r_B = np.array([0.005, 0.0, 0.0])
        self.I = np.array([30.0, 30.0, 30.0, 5.0, 15.0, 15.0])
        self.D = np.array([50.0, 75.0, 80.0, 5.0, 15.0, 15.0])
        self.D2 = np.array([25.0, 30.0, 40.0, 5.0, 15.0, 15.0])
        self.Ma = np.array([15.0, 40.0, 45.0, 5.0, 15.0, 15.0])

        self._dyn = Dynamics(self.m, self.v, self.rho, self.r_G, self.r_B, self.I, self.D, self.D2, self.Ma)

    def run(self, axis: int):
        self._curr_axis = axis

        csv_file = f'{self._data_dir}/{self._axes[axis]}-calib.csv'
        print(f'reading {csv_file}')
        self._d = pd.read_csv(csv_file, sep=',')

        time = self._d[['__time']].to_numpy()

        vel_field = self._vel_fields[axis]
        acc_field = self._acc_fields[axis]
        thrust_field = self._thrust_fields[axis]
        print(vel_field, acc_field, thrust_field)

        roll = self._d[['/gnc/pose/orientation/x']].to_numpy()
        roll = roll[np.logical_not(np.isnan(roll))]
        pitch = self._d[['/gnc/pose/orientation/y']].to_numpy()
        pitch = pitch[np.logical_not(np.isnan(pitch))]
        yaw = self._d[['/gnc/pose/orientation/z']].to_numpy()
        yaw = yaw[np.logical_not(np.isnan(yaw))]

        vel = self._d[[vel_field]].to_numpy()
        gnc_time = time[np.logical_not(np.isnan(vel))]
        vel = vel[np.logical_not(np.isnan(vel))]

        if axis in [0, 1, 2]:
            acc = self._d[[acc_field]].to_numpy()
            acc = acc[np.logical_not(np.isnan(acc))]
        else:
            acc = np.zeros(len(gnc_time))

            for i in range(len(gnc_time) - 1):
                if gnc_time[i + 1] != gnc_time[i]:
                    acc[i] = (vel[i + 1] - vel[i]) / (gnc_time[i + 1] - gnc_time[i])
                else:
                    acc[i] = acc[i - 1]

        if axis == 2:
            vel = -vel
            acc = -acc

        force = self._d[[thrust_field]].to_numpy()
        force_time = time[np.logical_not(np.isnan(force))]
        force = force[np.logical_not(np.isnan(force))]

        exp_force = np.zeros(len(gnc_time), np.float)
        for i in range(len(gnc_time)):
            eta = np.array([0.0, 0.0, 0.0, roll[i], pitch[i], yaw[i]])
            v = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            v[axis] = vel[i]
            v_d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            v_d[axis] = acc[i]

            tau = self._dyn.compute_tau(eta, v, v_d)

            exp_force[i] = tau[axis]

        fig = plt.figure(constrained_layout=True)
        # subfigs = fig.subfigures(1, 2, width_ratios=[3.0, 1.0])
        plt_axs = fig.subplots(3)
        plt_axs[0].plot(gnc_time, vel)
        plt_axs[1].plot(gnc_time, acc)
        plt_axs[2].plot(force_time, force)
        plt_axs[2].plot(gnc_time, exp_force)
        # txt_axs = subfigs[1].subplots(4)
        # I_tb = TextBox(txt_axs[0], 'I', np.array2string(self.I))
        # D_tb = TextBox(txt_axs[1], 'D', np.array2string(self.D))
        # D2_tb = TextBox(txt_axs[2], 'D2', np.array2string(self.D2))
        # Ma_tb = TextBox(txt_axs[3], 'Ma', np.array2string(self.Ma))
        plt.show()

def main():
    d = DynamicsTuner()
    for i in range(6):
        d.run(i)