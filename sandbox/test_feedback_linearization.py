

import torch
import numpy as np

from feedback_linearization import FeedbackLinearization
from double_pendulum import DoublePendulum

dyn = DoublePendulum(1.0, 1.0, 1.0, 1.0)
fb = FeedbackLinearization(dyn, 3, 5, torch.nn.ReLU(), 0.1)

u = np.array([1.0, 1.0])
x = np.array([1.0, 1.0, 1.0, 1.0])
v = np.array([1.0, 1.0])

M1 = fb._M1(x)
print("m1: ", M1)

#M2 = fb._M2(x)
#print("m2: ", M2)

w1 = fb._w1(x)
print("w1: ", w1)

#w2 = fb._w2(x)
#print("w2: ", w2)

mu = fb.feedback(x, v)
nu = fb.noisy_feedback(x, v)
print("mean u = ", mu)
print("noisy u = ", nu)

lp = fb.log_prob(u, x, v)
print("log prob of [1, 1] is ", lp)
