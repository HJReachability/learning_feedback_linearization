import torch
import numpy as np

from double_pendulum import DoublePendulum
from reinforce import Reinforce
from feedback_linearization import FeedbackLinearization

# Create a double pendulum.
mass1 = 1.0
mass2 = 1.0
length1 = 1.0
length2 = 1.0
time_step = 0.05
dyn = DoublePendulum(mass1, mass2, length1, length2, time_step)

# Create a feedback linearization object.
num_layers = 1
num_hidden_units = 5
activation = torch.nn.ReLU()
noise_std = 0.1
fb = FeedbackLinearization(
    dyn, num_layers, num_hidden_units, activation, noise_std)

# Create an initial state sampler for the double pendulum.
def initial_state_sampler():
    lower = np.array([[-1.0], [-0.1], [-1.0], [-0.1]])
    upper = -lower
    return np.random.uniform(lower, upper)

# Create REINFORCE.
num_iters = 10
learning_rate = 0.01
discount_factor = 0.99
num_rollouts = 5
num_steps_per_rollout = 10
solver = Reinforce(num_iters,
                   learning_rate,
                   discount_factor,
                   num_rollouts,
                   num_steps_per_rollout,
                   dyn,
                   initial_state_sampler,
                   fb)

# Run this guy.
solver.run()
