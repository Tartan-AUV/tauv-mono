import numpy as np
from math import sin, cos, tan

'''
function [dx, y] = model_linear_damping(t, x, u, L, I, m, v, CG, CB, varargin)
    dx = []; % static model

    rpy = u(1, 1:3);
    lin_v = u(1, 4:6);
    ang_v = u(1, 7:9);
    tau = u(1, 10:15);

    I_t = buildInertiaTensor(I(1), I(2), I(3), 0, 0, 0);
    M_rb = buildMassMatrix(m, CG, I_t);
    C_rb = buildCoriolisMatrix(m, CG, I_t, lin_v.', ang_v.');
    D_lin = diag([L]);
    G = buildGravityMatrix(m, v, CG, CB, rpy.');

    y = (M_rb \ (tau.' - G - C_rb * [lin_v.'; ang_v.'] - D_lin * [lin_v.'; ang_v.'])).';
end
'''

'''
function I = buildInertiaTensor(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    % Eq 4.3b: (Chin 2013, p 132)
    I = [Ixx, -Ixy, -Ixz;
        -Ixy,  Iyy, -Iyz;
        -Ixz, -Iyz,  Izz];
end
'''
def I(M_I: np.array) -> np.array:
    I = np.array([
       [M_I[0], -M_I[3], -M_I[4]],
       [-M_I[3], M_I[1], -M_I[5]],
       [-M_I[4], -M_I[5], M_I[2]]
    ])
    return I

'''
function M_rb = buildMassMatrix(m, r_G, I)
    % Eq 4.6: (Chin 2013, p 132)
    M_rb = [m * eye(3), -m*skew(r_G);
            m * skew(r_G), I];
end
'''
def M_rb(m: float, r_G: np.array, I: np.array) -> np.array:
    M_rb = np.block([
       [m * np.identity(3), -m * skew(r_G)],
       [m * skew(r_G), I]
    ])
    return M_rb

'''
function C_rb = buildCoriolisMatrix(m, r_G, I, v1, v2)
    % Eq 4.10: (Chin 2013, p 134)
    C_rb = [zeros(3),   -m*skew(v1) - m*skew(v2)*skew(r_G);
            -m*skew(v1) + skew(r_G)*skew(v2),   -skew(I*v2)];
end
'''
def C_rb(m: float, v: np.array, r_G: np.array, I: np.array) -> np.array:
    v_lin = v[0:3]
    v_ang = v[3:6]
    C_rb = np.block([
        [np.zeros((3, 3)), -m * skew(v_lin) - m * skew(v_ang) @ skew(r_G)],
        [-m * skew(v_lin) + skew(v_ang) @ skew(r_G), -skew(I @ v_ang)]
    ])
    return C_rb

'''
function [M_added, C_added] = buildAddedMassCoriolisMatrices(v, Ma_x, Ma_y, Ma_z, Ma_yaw, Ma_pitch, Ma_roll)
    % eq 4.32: (Chin 2013, p 143)
    M_added = -diag([Ma_x, Ma_y, Ma_z, Ma_yaw, Ma_pitch, Ma_roll]);
    
    % eq 4.33: (Chin 2013, p 143)
    % TODO: verify correctness!
    C_added = ...
        [0, 0, 0, 0, -Ma_z * v(3), Ma_y * v(2);
         0, 0, 0, Ma_z * v(3), 0, -Ma_x * v(1);
         0, 0, 0, -Ma_y * v(2), Ma_x * v(1), 0;
         0, -Ma_z * v(3), Ma_y * v(2), 0, -Ma_roll * v(6), Ma_pitch * v(5);
         Ma_z * v(3), 0, -Ma_x * v(1), Ma_roll * v(6), 0, -Ma_yaw * v(4);
         -Ma_y * v(2), Ma_x * v(1), 0, -Ma_pitch * v(5), Ma_yaw * v(4), 0];
end
'''
def M_a(v: np.array, Ma: np.array) -> np.array:
    M_a = -np.diag(Ma) * v
    return M_a

def C_a(v: np.array, Ma: np.array) -> np.array:
    C_a = np.array([
        [0, 0, 0, 0, -Ma[2] * v[2], Ma[1] * v[1]],
        [0, 0, 0, Ma[2] * v[3], 0, -Ma[0] * v[0]],
        [0, 0, 0, -Ma[1] * v[1], Ma[0] * v[0], 0],
        [0, -Ma[2] * v[2], Ma[1] * v[1], 0, -Ma[5] * v[5], Ma[4] * v[4]],
        [Ma[2] * v[2], 0, -Ma[0] * v[0], Ma[5] * v[5], 0, -Ma[3] * v[3]],
        [-Ma[1] * v[1], Ma[0] * v[0], 0, -Ma[4] * v[4], Ma[3] * v[3], 0]
    ])
    return C_a

'''
function G = buildGravityMatrix(m, b, r_G, r_B, n2)
    g = 9.81;
    W = m*g;
    B = b*g;
    
    phi = n2(1);
    theta = n2(2);
    cph = cos(phi);
    sph = sin(phi);
    cth = cos(theta);
    sth = sin(theta);
    
    x_G = r_G(1);
    y_G = r_G(2);
    z_G = r_G(3);
    x_B = r_B(1);
    y_B = r_B(2);
    z_B = r_B(3);
    
    % Eq 4.54 (Chin 2013, p 180)
    G = [(W-B) * sth;
         -(W-B) * cth*sph;
         -(W-B) * cth*cph;
         -(y_G*W - y_B*B)*cth*cph + (z_G*W - z_B*B)*cth*sph;
         (z_G*W - z_B*B)*sth + (x_G*W - x_B*B)*cth*cph;
         -(x_G*W - x_B*B)*cth*sph - (y_G *W - y_B*B)*sth;];
'''
def G(m: float, b: float, r_G: np.array, r_B: np.array, rpy: np.array) -> np.array:
    g = 9.81
    W = m * g
    B = b * g

    cphi = cos(rpy[0])
    sphi = sin(rpy[1])
    ctheta = cos(rpy[1])
    stheta = sin(rpy[1])

    G = np.array([
        (W - B) * stheta,
        -(W - B) * ctheta * sphi,
        -(W - B) * ctheta * cphi,
        -(r_G[1] * W - r_B[1] * B) * ctheta * cphi + (r_G[2] * W - r_B[2] * B) * ctheta * sphi,
        (r_G[2] * W - r_B[2] * B) * stheta + (r_G[0] * W - r_B[0] * B) * ctheta * cphi,
        -(r_G[0] * W - r_B[0] * B) * ctheta * sphi - (r_G[1] * W - r_B[1] * B) * stheta,
    ])
    return G

'''
function S = skew(x)
    % Chin 2013, p 133.
    % Defined as S(x)*y = cross(x,y)
    S = [0 -x(3) x(2) ; x(3) 0 -x(1) ; -x(2) x(1) 0 ];
end
'''

def skew(x: np.array) -> np.array:
    S = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])
    return S