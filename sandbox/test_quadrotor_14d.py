import numpy as np
import matplotlib.pyplot as plt

from quadrotor_14d import Quadrotor14D

# Create a quadrotor.
mass = 1.0
Ix = 1.0
Iy = 1.0
Iz = 1.0
time_step = 0.1
dyn = Quadrotor14D(mass, Ix, Iy, Iz, time_step)

# Create an initial state sampler for the double pendulum.
def initial_state_sampler():
    lower = np.array([[-1.0, -1.0, -1.0,
                       -np.pi - 0.5, -np.pi * 0.25, -np.pi * 0.25,
#                       -.1, -.1, -.1,
                       -0.1, -0.1, -0.1,
                       -1.0, # This is the thrust acceleration - g.
                       -0.1, -0.1, -0.1, -0.1]]).T
    upper = -lower
    lower[9, 0] = (lower[9, 0] + 9.81) / mass
    upper[9, 0] = (upper[9, 0] + 9.81) / mass

    return np.random.uniform(lower, upper)


x0 = initial_state_sampler()
v = np.zeros((4, 1))

# Collect output trajectory for feedback linearizing control (i.e., applying
# v -> u -> \dot x -> x -> y).
x = x0.copy()
xs = [x]
ys = []
us = []
HORIZON = 100
for ii in range(HORIZON):
    u = dyn.feedback(x, v)
    x = dyn.integrate(x, u)
    us.append(u)
    xs.append(x)
    ys.append(dyn.observation(x))

# Collect output trajectory for linearized system (i.e., applying
# v -> \dot z -> z -> y).
A, B, C = dyn.linearized_system()
print("A: ", A)
print("B: ", B)
print("C: ", C)

z0 = dyn.linearized_system_state(x0)
z = z0.copy()
zs_lin = [z.copy()]
ys_lin = []
zdot = lambda z_ref : A @ z_ref + B @ v
for ii in range(HORIZON):
    k1 = dyn._time_step * zdot(z)
    k2 = dyn._time_step * zdot(z + 0.5 * k1)
    k3 = dyn._time_step * zdot(z + 0.5 * k2)
    k4 = dyn._time_step * zdot(z + k3)

    z += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    zs_lin.append(z.copy())
    ys_lin.append(C @ z)

# Make sure that the `ys` and `ys_lin` are close to each other.
for ii in range(HORIZON):
    print("norm at %d is %f" % (ii, np.linalg.norm(ys[ii][:3, :] - ys_lin[ii][:3, :])))

# Plot positions.
x_coords = [y[0, 0] for y in ys]
y_coords = [y[1, 0] for y in ys]
z_coords = [y[2, 0] for y in ys]
psi_coords = [y[3, 0] for y in ys]

x_coords_lin = [y[0, 0] for y in ys_lin]
y_coords_lin = [y[1, 0] for y in ys_lin]
z_coords_lin = [y[2, 0] for y in ys_lin]
psi_coords_lin = [y[3, 0] for y in ys_lin]

z1_lin = [z[4, 0] for z in zs_lin]
z2_lin = [z[5, 0] for z in zs_lin]
z3_lin = [z[6, 0] for z in zs_lin]
z4_lin = [z[7, 0] for z in zs_lin]

u1 = [u[0, 0] for u in us]
u2 = [u[1, 0] for u in us]
u3 = [u[2, 0] for u in us]
u4 = [u[3, 0] for u in us]

y = [dyn.linearized_system_state(x)[4, 0] for x in xs]
dy = [dyn.linearized_system_state(x)[5, 0] for x in xs]
ddy = [dyn.linearized_system_state(x)[6, 0] for x in xs]
dddy = [dyn.linearized_system_state(x)[7, 0] for x in xs]

plt.figure()
plt.plot(np.arange(HORIZON), x_coords, '*b', label='fb_lin_controller')
plt.plot(np.arange(HORIZON), x_coords_lin, '.r', label='linearized_system')
plt.legend()
plt.title("x")

plt.figure()
plt.plot(np.arange(HORIZON), y_coords, '*b', label='fb_lin_controller')
plt.plot(np.arange(HORIZON), y_coords_lin, '.r', label='linearized_system')
plt.legend()
plt.title("y")

plt.figure()
plt.plot(np.arange(HORIZON), z_coords, '*b', label='fb_lin_controller')
plt.plot(np.arange(HORIZON), z_coords_lin, '.r', label='linearized_system')
plt.legend()
plt.title("z")

plt.figure()
plt.plot(np.arange(HORIZON), psi_coords, '*b', label='fb_lin_controller')
plt.plot(np.arange(HORIZON), psi_coords_lin, '.r', label='linearized_system')
plt.legend()
plt.title("phi")

plt.figure()
plt.plot(np.arange(HORIZON), u1, '*r', label='u1')
plt.plot(np.arange(HORIZON), u2, '*g', label='u2')
plt.plot(np.arange(HORIZON), u3, '*b', label='u3')
plt.plot(np.arange(HORIZON), u4, '*k', label='u4')
plt.legend()
plt.title("u")

plt.figure()
plt.plot(np.arange(HORIZON + 1), y, '*r', label='y')
plt.plot(np.arange(HORIZON + 1), z1_lin, '.r', label='z1_lin')
plt.plot(np.arange(HORIZON + 1), dy, '*g', label='dy')
plt.plot(np.arange(HORIZON + 1), z2_lin, '.g', label='z2_lin')
plt.plot(np.arange(HORIZON + 1), ddy, '*b', label='ddy')
plt.plot(np.arange(HORIZON + 1), z3_lin, '.b', label='z3_lin')
plt.plot(np.arange(HORIZON + 1), dddy, '*k', label='dddy')
plt.plot(np.arange(HORIZON + 1), z4_lin, '.k', label='z4_lin')
plt.legend()
plt.title("dy")






plt.show()
