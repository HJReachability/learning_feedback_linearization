import torch
import numpy as np
import time

from double_pendulum import DoublePendulum
from reinforce import Reinforce
from ppo import PPO
from feedback_linearization import FeedbackLinearization
from logger import Logger

# Seed everything.
seed = np.random.choice(1000)
torch.manual_seed(seed)
np.random.seed(seed)

# Create a double pendulum.
mass1 = 1.0
mass2 = 1.0
length1 = 1.0
length2 = 1.0
time_step = 0.02
friction_coeff = 0.5
dyn = DoublePendulum(mass1, mass2, length1, length2, time_step, friction_coeff)

# Bad dynamics guess or train from zero


mass1_scaling = 0.66
mass2_scaling = 0.66
length1_scaling = 1.0
length2_scaling = 1.0
friction_coeff_scaling = 1.0
bad_dyn = DoublePendulum(
    mass1_scaling * mass1, mass2_scaling * mass2,
    length1_scaling * length1, length2_scaling * length2,
    time_step, friction_coeff_scaling * friction_coeff)

# Create a feedback linearization object.
num_layers = 2
num_hidden_units = 32
activation = torch.nn.Tanh()
noise_std = 1.0
fb = FeedbackLinearization(
    bad_dyn, num_layers, num_hidden_units, activation, noise_std)


# Choose Algorithm
do_PPO=0
do_Reinforce=1

#Algorithm Params ** Only for Reinforce:

## Train for zero (no bad dynamics)
from_zero=1

# Rewards scaling - default is 10.0
scale_rewards=100.0

# norm to use
norm=2

if from_zero:
	fb._M1= lambda x : np.zeros((2,2))
	fb._f1= lambda x : np.zeros((2,1))


assert do_PPO!=do_Reinforce

# Create an initial state sampler for the double pendulum.
def initial_state_sampler(num):

	if num<0000:
		lower = (1.0 - num/1000.0)*np.array([[-0.5 ], [-0.5], [-0.5], [-0.5]]) + num/1000.0*np.array([[-np.pi], [-0.5], [-np.pi], [-0.5]])
		upper = -lower
	else:
		lower = np.array([[-np.pi], [-0.5], [-np.pi], [-0.5]])
		upper = -lower

	# lower = np.array([[-np.pi], [-0.5], [-np.pi], [-0.5]])
	# upper = -lower

	return np.random.uniform(lower, upper)

# Create Solver.
num_iters = 2000
learning_rate = 1e-3
desired_kl = -1.0
discount_factor = 0.99
num_rollouts = 50
num_steps_per_rollout = 25


# Logging.


if do_PPO:

	logger = Logger(
    "./logs/double_pendulum_PPO_%dx%d_std%f_lr%f_kl%f_%d_%d_dyn_%f_%f_%f_%f_%f_seed_%d.pkl" %
    (num_layers, num_hidden_units, noise_std, learning_rate, desired_kl,
     num_rollouts, num_steps_per_rollout,
     mass1_scaling, mass2_scaling,
     length1_scaling, length2_scaling, friction_coeff_scaling,
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
    "./logs/double_pendulum_Reinforce_%dx%d_std%f_lr%f_kl%f_%d_%d_norm%f_dyn_%f_%f_%f_%f_%f_seed_%d.pkl" %
    (num_layers, num_hidden_units, noise_std, learning_rate, desired_kl,
     num_rollouts, num_steps_per_rollout,norm,
     mass1_scaling, mass2_scaling,
     length1_scaling, length2_scaling, friction_coeff_scaling,
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
	             scale_rewards)


# Run this guy.
solver.run(plot=False,show_diff=False)

# Dump the log.
logger.dump()
