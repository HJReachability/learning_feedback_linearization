import torch
import numpy as np
from feedback_linearization import FeedbackLinearization
from reinforce import Reinforce
from quadrotor_14d import Quadrotor14D

dyn = Quadrotor14D(1.0, 1.0, 1.0, 1.0, 0.01)
fb = FeedbackLinearization(dyn, 2, 32, torch.nn.Tanh(), 0.1)
solver = Reinforce(1, 1, 1, 1, 1, 10000, dyn, None, fb, None, 1, 1, None)

x0 = np.zeros((14, 1))
#x0[0, 0] = x0[1, 0] = 0.1
#x0[2, 0] = -0.1
x0[9, 0] = 9.81

ref, K = solver._generate_reference()
ys, xs = solver._generate_ys(x0, ref, K)

import matplotlib.pyplot as plt
plt.figure()
plt.plot([x[0, 0] for x in xs], label="x")
plt.plot([x[1, 0] for x in xs], label="y")
plt.plot([x[3, 0] for x in xs], label="yaw")
plt.plot([x[4, 0] for x in xs], label="pitch")
plt.plot([x[5, 0] for x in xs], label="roll")

plt.plot([r[0, 0] for r in ref], label="x_ref")
plt.plot([r[4, 0] for r in ref], label="y_ref")
plt.plot([r[12, 0] for r in ref], label="yaw_ref")

plt.legend()
plt.show()
