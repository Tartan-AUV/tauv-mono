# dynamics.py
#
# AUV Dynamics model, intended for use as inverse dynamics for differential flatness control
# Goal is to provide tau as a function of the flat outputs. See paper for details:
#
# G. Rigatos et al (2017):
# AUV Control and Navigation with Differential Flatness Theory
# and Derivative-Free Nonlinear Kalman Filtering
#
# Important note: Dynamics are modelled in the NED frame!

import numpy as np
from math import sin, cos, tan


class Dynamics:
    def __init__(self):
        # TODO: load these from a yaml file in tauv_config (or better, the URDF itself!)
        self.m = 15  # mass of vehicle
        self.b = 18.5  # mass of displaced water (b = volume * rho)
        self.r_G = [0, 0, 0]  # defined in body NED frame!
        self.r_B = [0, 0, 0]  # defined in body NED frame!

        self.Ixx = 5
        self.Iyy = 5
        self.Izz = 10
        self.Ixy = 0
        self.Ixz = 0
        self.Iyz = 0

        # See eq 11 for derivation from damping coefficients:
        self.D_x = 10
        self.D_y = 10
        self.D_z = 40
        self.D_roll = 13
        self.D_pitch = 13
        self.D_yaw = 1

        # Todo: add quadratic damping to the model
        # self.D2_x = 0
        # self.D2_y = 0
        # self.D2_z = 0
        # self.D2_roll = 0
        # self.D2_pitch = 0
        # self.D2_yaw = 0

        self.Ma_x = 10
        self.Ma_y = 10
        self.Ma_z = 50
        self.Ma_roll = 1
        self.Ma_pitch = 1
        self.Ma_yaw = 0

    def get_eta_d(self, eta, v):
        return np.dot(self._J(eta), v)

    def get_v(self, eta, eta_d):
        return np.dot(np.linalg.inv(self._J(eta)), eta_d)

    def get_eta_dd(self, eta, eta_d, v_d):
        J = self._J(eta)
        J_d = self._J_d(eta, eta_d)
        J_inv = np.linalg.inv(J)

        a = np.dot(np.dot(J_inv, J_inv), np.dot(J_d, eta_d))
        b = v_d + a
        return np.dot(J, b)

    def get_v_d(self, eta, eta_d, eta_dd):
        J = self._J(eta)
        J_d = self._J_d(eta, eta_d)
        J_inv = np.linalg.inv(J)

        a = np.dot(J_inv, eta_dd)
        b = np.dot(np.dot(J_inv, J_inv), np.dot(J_d, eta_d))
        return a - b

    def compute_tau(self, eta, eta_d, eta_dd):
        J = self._J(eta)
        tau_n = self.compute_tau_n(eta, eta_d, eta_dd)
        tau = np.dot(J.T, tau_n)
        return tau

    def compute_tau_n(self, eta, eta_d, eta_dd):
        # eta = measured state [x,y,z,phi,theta,psi]
        # eta_d = measured derivatives of eta
        # eta_dd = desired fixed frame acceleration

        eta = np.array(eta)
        eta_d = np.array(eta_d)
        eta_dd = np.array(eta_dd)

        v = self.get_v(eta, eta_d)

        M_rb = self._M_rb()
        C_rb = self._C_rb(v)
        M_a = self._M_a()
        C_a = self.C_a(v)
        D = self._D(v)
        G = self._G(eta)

        M = M_rb + M_a
        C = C_rb + C_a

        J = self._J(eta)
        J_d = self._J_d(eta, eta_d)
        J_inv = np.linalg.inv(J)
        J_tinv = J_inv.T

        M_n = np.dot(np.dot(J_tinv, M), J_inv)
        C_n = np.dot(np.dot(J_tinv, C - np.dot(np.dot(M, J_inv), J_d)), J_inv)
        D_n = np.dot(np.dot(J_tinv, D), J_inv)
        G_n = np.dot(J_tinv, G)

        tau_n = np.dot(M_n, eta_dd) + np.dot(C_n, eta_d) + np.dot(D_n, eta_d) + G_n.T
        return tau_n.T

    ######################
    # Dynamics Matrices: #
    ######################

    def _M_rb(self):
        # Mass matrix in body frame
        # Eq 5
        m = self.m
        xg = self.r_G[0]
        yg = self.r_G[1]
        zg = self.r_G[2]
        Ix = self.Ixx
        Iy = self.Iyy
        Iz = self.Izz
        Ixy = self.Ixy
        Ixz = self.Ixz
        Iyz = self.Iyz

        return np.array([
            [m, 0, 0, 0, m * zg, -m * yg],
            [0, m, 0, -m * zg, 0, m * xg],
            [0, 0, m, m * yg, -m * xg, 0],
            [0, -m * zg, m * yg, Ix, -Ixy, -Ixz],
            [m * zg, 0, -m * xg, -Ixy, Iy, -Iyz],
            [-m * yg, m * xg, 0, -Ixz, -Iyz, Iz]
        ])

    def _C_rb(self, vel):
        # Coriolis matrix in body frame
        # vel is velocity in the body frame (vel = [u1, u2])
        # eq 6 (Note typo in C_rb(3,6): negative shouldn't be there.)

        m = self.m
        xg = self.r_G[0]
        yg = self.r_G[1]
        zg = self.r_G[2]
        Ix = self.Ixx
        Iy = self.Iyy
        Iz = self.Izz
        Ixy = self.Ixy
        Ixz = self.Ixz
        Iyz = self.Iyz
        u = vel[0]
        v = vel[1]
        w = vel[2]
        p = vel[3]
        q = vel[4]
        r = vel[5]

        C_12 = np.array([
            [m * (yg * q + zg * r), -m * (xg * q - w), -m * (xg * r + v)],
            [-m * (yg * p + w), m * (zg * r + xg * p), -m * (yg * r - u)],
            [-m * (zg * p - v), -m * (zg * q + u), m * (xg * p + yg * q)],
        ])
        C_22 = np.array([
            [0, -Iyz * q - Ixz * p + Iz * r, Iyz * r + Ixy * p - Iy * q],
            [Ixy * q + Ixz * p - Iz * r, 0, Ixz * r + Ixy * q + Ix * p],
            [-Iyz * r - Ixy * p - Iy * q, Ixz * r + Ixy * q - Ix * p, 0]
        ])
        return np.vstack((
            np.hstack((np.zeros((3,3)), C_12)),
            np.hstack((-C_12.T, C_22))
        ))

    def _M_a(self):
        # Added mass mass matrix
        # Eq 8
        a1 = self.Ma_x
        a2 = self.Ma_y
        a3 = self.Ma_z
        a4 = self.Ma_roll
        a5 = self.Ma_pitch
        a6 = self.Ma_yaw

        return np.diag([a1, a2, a3, a4, a5, a6])

    def C_a(self, vel):
        # Added mass coriolis matrix
        # Eq 9

        a1 = self.Ma_x
        a2 = self.Ma_y
        a3 = self.Ma_z
        a4 = self.Ma_roll
        a5 = self.Ma_pitch
        a6 = self.Ma_yaw
        u = vel[0]
        v = vel[1]
        w = vel[2]
        p = vel[3]
        q = vel[4]
        r = vel[5]

        return np.array([
            [0, 0, 0, 0, a3 * w, -a2 * v],
            [0, 0, 0, -a3 * w, 0, a1 * u],
            [0, 0, 0, a2 * v, -a1 * u, 0],
            [0, a3 * w, -a2 * v, 0, a6 * r, -a5 * q],
            [-a3 * w, 0, a1 * u, -a6 * r, 0, a4 * p],
            [a2 * v, -a1 * u, 0, a5 * q, -a4 * p, 0]
        ])

    def _D(self, vel):
        # damping matrix
        # Eq 10

        X = self.D_x
        Y = self.D_y
        Z = self.D_z
        K = self.D_roll
        M = self.D_pitch
        N = self.D_yaw
        u = vel[0]
        v = vel[1]
        w = vel[2]
        p = vel[3]
        q = vel[4]
        r = vel[5]

        return np.diag([X * abs(u), Y * abs(v), Z * abs(w), K * abs(p), M * abs(q), N * abs(r)])

    def _G(self, x):
        # Gravity compensation vector:
        # Eq 12
        # x is vehicle state (in world fixed NED frame)

        xg = self.r_G[0]
        yg = self.r_G[1]
        zg = self.r_G[2]
        xb = self.r_B[0]
        yb = self.r_B[1]
        zb = self.r_B[2]
        W = self.m * 9.81
        B = self.b * 9.81
        phi = x[3]
        theta = x[4]

        return np.array([
            [(W - B) * sin(theta)],
            [-(W - B) * cos(theta) * sin(phi)],
            [-(W - B) * cos(theta) * cos(phi)],
            [-(yg * W - yb * B) * cos(theta) * cos(phi) + (zg * W - zb * B) * cos(theta) * sin(phi)],
            [(zg * W - zb * B) * sin(theta) + (xg * W - xb * B) * cos(theta) * cos(phi)],
            [-(xg * W - xb * B) * cos(theta) * sin(phi) - (yg * W - yb * B) * sin(theta)]
        ])

    def _J(self, x):
        # fixed <-> body frame transformation matrix
        # Eq 1 and Eq 2
        # eta = J * v (eta is fixed frame velocity, v is body frame velocity)
        # x is vehicle state (in world fixed NED frame)
        phi = x[3]
        theta = x[4]
        psi = x[5]

        cphi = cos(phi)
        sphi = sin(phi)
        cth = cos(theta)
        sth = sin(theta)
        tth = tan(theta)
        cpsi = cos(psi)
        spsi = sin(psi)

        # Prevent errors at singularity:
        if abs(cth) < 0.0001:
            cth = np.sign(cth) * 0.0001

        J1 = np.array([
            [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
            [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cth],
            [-sth, cth * sphi, cth * cphi]
        ])

        J2 = np.array([
            [1, sphi * tth, cphi * tth],
            [0, cphi, -sphi],
            [0, sphi / cth, cphi / cth]
        ])

        return np.vstack((
            np.hstack((J1, np.zeros((3,3)))),
            np.hstack((np.zeros((3,3)), J2))
        ))

    def _J_d(self, eta, eta_d):
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]
        dphidt = eta_d[3]
        dthetdt = eta_d[4]
        dpsidt = eta_d[5]

        dTdphi = np.array([
            [0, cos(phi) * tan(theta), -sin(phi) * tan(theta)],
            [0, -sin(phi), -cos(phi)],
            [0, cos(phi) / cos(theta), -sin(phi) / cos(theta)]
        ])

        dTdtheta = np.array([
            [0, sin(phi) * (tan(theta) ** 2 + 1), cos(phi) * (tan(theta) ** 2 + 1)],
            [0, 0, 0],
            [0, (sin(phi) * sin(theta)) / cos(theta) ** 2, (cos(phi) * sin(theta)) / cos(theta) ** 2]
        ])

        dRdphi = np.array([
            [0, sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta),
             cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)],
            [0, cos(phi) * sin(psi) * sin(theta) - cos(psi) * sin(phi), -cos(phi) * cos(psi)],
            [0, cos(phi) * cos(theta), -cos(theta) * sin(phi)]
        ])

        dRdtheta = np.array([
            [-cos(psi) * sin(theta), cos(psi) * cos(theta) * sin(phi), cos(phi) * cos(psi) * cos(theta)],
            [-sin(psi) * sin(theta), cos(theta) * sin(phi) * sin(psi), cos(theta) ** 2 * sin(psi) - sin(psi) * sin(
                theta) ** 2],
            [-cos(theta), -sin(phi) * sin(theta), -cos(phi) * sin(theta)]
        ])

        dRdpsi = np.array([
            [-cos(theta) * sin(psi), - cos(phi) * cos(psi) - sin(phi) * sin(psi) * sin(theta),
             cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)],
            [cos(psi) * cos(theta), cos(psi) * sin(phi) * sin(theta) - cos(phi) * sin(psi), sin(phi) * sin(psi) + cos(
                psi) * cos(theta) * sin(theta)],
            [0, 0, 0]
        ])

        dRdt = dRdphi * dphidt + dRdtheta * dthetdt + dRdpsi * dpsidt
        dTdt = dTdphi * dphidt + dTdtheta * dthetdt

        dJdt = np.vstack((
            np.hstack((dRdt, np.zeros((3,3)))),
            np.hstack((np.zeros((3,3)), dTdt)))
        )

        return dJdt

    def _skew(self, x):
        return np.array([
            [0, -x[2], x[1]],
            [x[2], 0, -x[0]],
            [-x[1], x[0], 0]
        ])
