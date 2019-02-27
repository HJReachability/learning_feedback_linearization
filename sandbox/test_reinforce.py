import torch
import numpy as np

from double_pendulum import DoublePendulum
from reinforce import Reinforce
from feedback_linearization import FeedbackLinearization
from logger import Logger

# Create a double pendulum.
mass1 = 1.0
mass2 = 1.0
length1 = 1.0
length2 = 1.0
time_step = 0.02
friction_coeff = 0.5
dyn = DoublePendulum(mass1, mass2, length1, length2, time_step, friction_coeff)

mass1_scaling = 1.05
mass2_scaling = 0.95
length1_scaling = 1.0
length2_scaling = 1.0
bad_dyn = DoublePendulum(
    mass1_scaling * mass1, mass2_scaling * mass2,
    length1_scaling * length1, length2_scaling * length2,
    time_step, friction_coeff)

# Create a feedback linearization object.
num_layers = 3
num_hidden_units = 10
activation = torch.nn.Tanh()
noise_std = 0.1
fb = FeedbackLinearization(
    bad_dyn, num_layers, num_hidden_units, activation, noise_std)

# Create an initial state sampler for the double pendulum.
def initial_state_sampler():
    lower = np.array([[-1.0], [-0.1], [-1.0], [-0.1]])
    upper = -lower
    return np.random.uniform(lower, upper)


# Create REINFORCE.
num_iters = 2000
learning_rate = 1e-3
discount_factor = 0.99
num_rollouts = 50
num_steps_per_rollout = 25

# Logging.
logger = Logger("logs/double_pendulum_%d_%d_%f_%f_%d_%d_dyn_%f_%f_%f_%f.pkl" %
                (num_layers, num_hidden_units, noise_std, learning_rate,
                 num_rollouts, num_steps_per_rollout,
                 mass1_scaling, mass2_scaling,
                 length1_scaling, length2_scaling))

solver = Reinforce(num_iters,
                   learning_rate,
                   discount_factor,
                   num_rollouts,
                   num_steps_per_rollout,
                   dyn,
                   initial_state_sampler,
                   fb,
                   logger)

# Run this guy.
solver.run(plot=False)

# Dump the log.
logger.dump()
