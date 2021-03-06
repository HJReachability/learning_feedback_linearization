import torch
import numpy as np
import time

from constraint import Constraint
from quadrotor_12d import Quadrotor12D
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
time_step = 0.01
dyn = Quadrotor12D(mass, Ix, Iy, Iz, time_step)

mass_scaling = 0.75
Ix_scaling = 0.5
Iy_scaling = 0.5
Iz_scaling = 0.5
bad_dyn = Quadrotor12D(
    mass_scaling * mass, Ix_scaling * Ix,
    Iy_scaling * Iy, Iz_scaling * Iz, time_step)

# Create a feedback linearization object.
num_layers = 2
num_hidden_units = 32
activation = torch.nn.Tanh()
noise_std = 0.25
fb = FeedbackLinearization(
    bad_dyn, num_layers, num_hidden_units, activation, noise_std)


# Choose Algorithm
do_PPO=0
do_Reinforce=1

# Create an initial state sampler for the double pendulum.
def initial_state_sampler(num):
    lower0 = np.array([[-0.25, -0.25, -0.25,
                       -0.1, -0.1,
                       -0.1, -0.1, -0.1,
                       -1.0, # This is the thrust acceleration - g.
                       -0.1, -0.1, -0.1]]).T
    lower1 = np.array([[-2.5, -2.5, -2.5,
                        -np.pi / 4.0, -np.pi / 4.0,
                       -0.3, -0.3, -0.3,
                       -3.0, # This is the thrust acceleration - g.
                       -0.3, -0.3, -0.3]]).T

    frac = 1.0 #min(float(num) / 1500.0, 1.0)
    lower = frac * lower1 + (1.0 - frac) * lower0
    upper = -lower

    lower[8, 0] = (lower[8, 0] + 9.81) / mass
    upper[8, 0] = (upper[8, 0] + 9.81) / mass

    return np.random.uniform(lower, upper)

# Create REINFORCE.
num_iters = 3000
learning_rate = 1e-4
desired_kl = -1.0
discount_factor = 1.0
num_rollouts = 25
num_steps_per_rollout = 100

# Constraint on state so that we don't go nuts.
class Quadrotor12DConstraint(Constraint):
    def contains(self, x):
        BIG = 100.0
        SMALL = 0.01
        return abs(x[0, 0]) < BIG and \
            abs(x[1, 0]) < BIG and \
            abs(x[2, 0]) < BIG and \
            abs(x[3, 0]) < np.pi / 2.5 and \
            abs(x[4, 0]) < np.pi / 2.5 and \
            abs(x[5, 0]) < BIG and \
            abs(x[6, 0]) < BIG and \
            abs(x[7, 0]) < BIG and \
            abs(x[8, 0]) > SMALL and \
            abs(x[9, 0]) < BIG and \
            abs(x[10, 0]) < BIG and \
            abs(x[11, 0]) < BIG

state_constraint = Quadrotor12DConstraint()

#Algorithm Params ** Only for Reinforce:
## Train for zero (no bad dynamics)
from_zero=False

# Rewards scaling - default is 10.0
scale_rewards=10.0

# norm to use
norm=2

if from_zero:
    fb._M1= lambda x : np.zeros((4,4))
    fb._f1= lambda x : np.zeros((4,1))

if do_PPO:
    logger = Logger(
        "logs/quadrotor_12d_PPO_%dx%d_std%f_lr%f_kl%f_%d_%d_dyn_%f_%f_%f_%f_seed_%d.pkl" %
        (num_layers, num_hidden_units, noise_std, learning_rate, desired_kl,
         num_rollouts, num_steps_per_rollout,
         mass_scaling, Ix_scaling, Iy_scaling, Iz_scaling,
         seed))
    solver = PPO(num_iters,
                 learning_rate,
                 desired_kl,
                 discount_factor,
                 num_rollouts,
                 num_steps_per_rollout,
                 dyn,
                 initial_state_sampler,
                 fb,
                 logger)

if do_Reinforce:
    logger = Logger(
        "logs/quadrotor_12d_Reinforce_%dx%d_std%f_lr%f_kl%f_%d_%d_fromzero_%s_dyn_%f_%f_%f_%f_seed_%d_norm_%d_smallweights_tanh.pkl" %
        (num_layers, num_hidden_units, noise_std, learning_rate, desired_kl,
         num_rollouts, num_steps_per_rollout, str(from_zero),
         mass_scaling, Ix_scaling, Iy_scaling, Iz_scaling,
         seed, norm))
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
                       scale_rewards,
                       state_constraint)

# Set number of threads.
torch.set_num_threads(1)

# Run this guy.
solver.run(plot=False, show_diff=False)

# Dump the log.
logger.dump()
