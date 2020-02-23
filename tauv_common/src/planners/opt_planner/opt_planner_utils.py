import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# State dims
class State3d():
    dim = 12
    x = 0
    y = 1
    z = 2
    roll = 3
    pitch = 4
    yaw = 5
    x_dot = 6
    y_dot = 7
    z_dot = 8
    roll_dot = 9
    pitch_dot = 10
    yaw_dot = 11

class Control3d():
    dim = 6
    a_x = 0
    a_y = 1
    a_z = 2
    a_roll = 3
    a_pitch = 4
    a_yaw = 5


def plot_3d_traj(states, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, State3d.x], states[:, State3d.y], states[:, State3d.z])

    # ax.quiver(states[:, State3d.x], states[:, State3d.y], states[:, State3d.z],
    #           states[:, State3d.roll], states[:, State3d.pitch], states[:, State3d.yaw])

    ax.scatter(states[0, State3d.x], states[0, State3d.y], states[0, State3d.z], marker='o', color='green', zorder=2)
    ax.scatter(states[-1, State3d.x], states[-1, State3d.y], states[-1, State3d.z],marker='o', color='red', zorder=2)
    if show:
        plt.show()
