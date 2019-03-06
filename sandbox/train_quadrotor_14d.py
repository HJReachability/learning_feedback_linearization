import torch
import numpy as np
import time

from quadrotor_14d import Quadrotor14D
from reinforce import Reinforce
from feedback_linearization import FeedbackLinearization
from logger import Logger

# Seed everything.
seed = np.random.choice(1000)
torch.manual_seed(seed)
np.random.seed(seed)

# Create a quadrotor.
mass = 1.0
Ix = 1.0
Iy = 1.0
Iz = 1.0
time_step = 0.02
dyn = Quadrotor14D(mass, Ix, Iy, Iz, time_step)

mass_scaling = 0.9
Ix_scaling = 1.1
Iy_scaling = 1.1
Iz_scaling = 1.1
bad_dyn = Quadrotor14D(
    mass_scaling * mass, Ix_scaling * Ix,
    Iy_scaling * Iy, Iz_scaling * Iz, time_step)

# Create a feedback linearization object.
num_layers = 3
num_hidden_units = 10
activation = torch.nn.Tanh()
noise_std = 1.0
fb = FeedbackLinearization(
    bad_dyn, num_layers, num_hidden_units, activation, noise_std)

# Create an initial state sampler for the double pendulum.
def initial_state_sampler(num):
    if num < 500:
        lower = np.array([[-0.25, -0.25, -0.25,
                           -0.1, -0.1, -0.1,
                           -0.1, -0.1, -0.1,
                           -1.0, # This is the thrust acceleration - g.
                           -0.1, -0.1, -0.1, -0.1]]).T
    elif num < 1500:
        lower = np.array([[-1.0, -1.0, -1.0,
                           -0.5, -0.25, -0.25,
                           -0.2, -0.2, -0.2,
                           -1.5, # This is the thrust acceleration - g.
                           -0.2, -0.2, -0.2, -0.2]]).T
    else:
        lower = np.array([[-2.5, -2.5, -2.5,
                           -np.pi, -1.0, -1.0,
                           -0.3, -0.3, -0.3,
                           -2.0, # This is the thrust acceleration - g.
                           -0.3, -0.3, -0.3, -0.3]]).T

    upper = -lower
    lower[9, 0] = (lower[9, 0] + 9.81) / mass
    upper[9, 0] = (upper[9, 0] + 9.81) / mass

    return np.random.uniform(lower, upper)

# Create REINFORCE.
num_iters = 2000
learning_rate = 1e-3
desired_kl = -1.0
discount_factor = 0.99
num_rollouts = 50
num_steps_per_rollout = 25
norm = 2
SCALING = 100.0

# Logging.
logger = Logger(
    "logs/quadrotor_14d_%dx%d_std%f_lr%f_kl%f_%d_%d_norm%d_dyn_%f_%f_%f_%f_seed_%d.pkl" %
    (num_layers, num_hidden_units, noise_std, learning_rate, desired_kl,
     num_rollouts, num_steps_per_rollout, norm,
     mass_scaling, Ix_scaling, Iy_scaling, Iz_scaling,
     seed))

solver = Reinforce(num_iters,
                   learning_rate,
                   desired_kl,
                   discount_factor,
                   num_rollouts,
                   num_steps_per_rollout,
                   dyn,
                   initial_state_sampler,
                   fb,
                   logger,
                   norm,
                   SCALING)

# Run this guy.
solver.run(plot=False)

# Dump the log.
logger.dump()
