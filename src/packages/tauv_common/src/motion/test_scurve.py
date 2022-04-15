import numpy as np
import matplotlib.pyplot as plt
from trajectories.scurve import SCurve


def main():
    q0 = 0.0
    q1 = 1.0
    v0 = 1.1
    v_max = 1.0
    a_max = 0.3
    j_max = 1.0

    p = SCurve(q1, v_max, a_max, j_max)
    (q, v, a, j) = p.plan(q0, v0)

    time = np.linspace(0, 10.0, 1000)
    q_list = np.array([q(t) for t in time])
    v_list = np.array([v(t) for t in time])
    a_list = np.array([a(t) for t in time])
    j_list = np.array([j(t) for t in time])

    fig, axs = plt.subplots(4)
    axs[0].plot(time, q_list)
    axs[1].plot(time, v_list)
    axs[2].plot(time, a_list)
    axs[3].plot(time, j_list)
    plt.show()


if __name__ == '__main__':
    main()