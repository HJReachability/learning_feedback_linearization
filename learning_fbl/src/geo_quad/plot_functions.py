import numpy as np
import matplotlib.pyplot as plt




def plot_state_trajectory(states, refs, dt):

    times = dt*np.array(range(min([len(states), len(refs)])))
    xs = []
    ys = []
    zs = []
    rxs = []
    rys = []
    rzs = []
    for i in range(min([len(states), len(refs)])):
        xs.append(states[i][0])
        ys.append(states[i][1])
        zs.append(states[i][2])
        rxs.append(refs[i][0])
        rys.append(refs[i][1])
        rzs.append(refs[i][2])

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    rxs = np.array(rxs)
    rys = np.array(rys)
    rzs = np.array(rzs)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(times, xs, label='Actual')
    plt.plot(times, rxs, label='Desired')
    plt.xlabel("Time (t)")
    plt.ylabel('x')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(times, ys, label='Actual')
    plt.plot(times, rys, label='Desired')
    plt.xlabel("Time (t)")
    plt.ylabel('y')
    plt.legend()

    plt.subplot(3,1,3)
    plt.plot(times, zs, label='Actual')
    plt.plot(times, rzs, label='Desired')
    plt.xlabel("Time (t)")
    plt.ylabel('z')
    plt.legend()

    plt.show()

        



















