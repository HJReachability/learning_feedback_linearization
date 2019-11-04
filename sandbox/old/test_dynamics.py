import torch
import numpy as np

from double_pendulum import DoublePendulum

dyn = DoublePendulum(1.0, 1.0, 1.0, 1.0, time_step=0.01)

u = np.array([[0.0], [0.0]])
x = np.array([[0.1], [0.0], [0.1], [0.0]])

current_x = x.copy()
for ii in range(10):
    current_x = dyn.integrate(current_x, u)
    print(current_x.T)
