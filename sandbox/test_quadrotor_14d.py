import numpy as np

from quadrotor_14d import Quadrotor14D

# Create a quadrotor.
mass = 1.0
Ix = 1.0
Iy = 1.0
Iz = 1.0
time_step = 0.02
dyn = Quadrotor14D(mass, Ix, Iy, Iz, time_step)

# Create an initial state sampler for the double pendulum.
def initial_state_sampler():
    lower = np.array([[-1.0, -1.0, -1.0,
                       -np.pi - 0.5, -np.pi * 0.25, -np.pi * 0.25,
                       -0.1, -0.1, -0.1,
                       -1.0, # This is the thrust acceleration - g.
                       -0.1, -0.1, -0.1, -0.1]]).T
    upper = -lower
    lower[9, 0] = (lower[9, 0] + 9.81) / mass
    upper[9, 0] = (upper[9, 0] + 9.81) / mass

    return np.random.uniform(lower, upper)


while True:
    x0 = initial_state_sampler()
    v = x0[:4, :]

    x = x0.copy()

    xs = [x0]
    us = []
    for ii in range(10):
        u = dyn.feedback(x, v)
        x = dyn.integrate(x, u)
        xs.append(x)
        us.append(u)
        if np.isnan(x).any():
            break

    if np.isnan(x).any():
        break

print(xs)
print("-----")
print(us)
