

import torch
import numpy as np

from feedback_linearization import FeedbackLinearization
from double_pendulum import DoublePendulum

dyn = DoublePendulum(1.0, 1.0, 1.0, 1.0, time_step=0.01)
fb = FeedbackLinearization(dyn, 3, 5, torch.nn.ReLU(), 0.1)

u = np.array([1.0, 1.0])
x = np.array([1.0, 1.0, 1.0, 1.0])

v = np.array([1.0, 1.0])

#M1 = fb._M1(x)
#print("m1: ", M1)

#M2 = fb._M2(x)
#print("m2: ", M2)

#w1 = fb._w1(x)\
#print("w1: ", w1)

#w2 = fb._w2(x)
#print("w2: ", w2)

#mu = fb.feedback(x, v)
#nu = fb.noisy_feedback(x, v)
#print("mean u = ", mu)
#print("noisy u = ", nu)

#lp = fb.log_prob(u, x, v)
#print("log prob of [1, 1] is ", lp)

# Generate v, y desired and check that we can match with no model mismatch.
from reinforce import Reinforce
r = Reinforce(1, 1, 1, 1, 100, dyn, None, fb)
current_x = np.array([[0.1], [0.0], [0.1], [0.0]])
vs = r._generate_v()
y_desireds = r._generate_y(current_x, vs)

ys = []
print("current x: ", current_x)
for v, y_desired in zip(vs, y_desireds):
    u = dyn._f_q(current_x) + dyn._M_q(current_x) @ v
    current_x = dyn.integrate(current_x, u, dt=0.001)
    ys.append(dyn.observation(current_x))

print("ys: ", ys)
print("y_desireds:", y_desireds)
