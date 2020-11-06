#!/usr/bin/env python

import numpy as np
from swingup_class import SwingupFBL
from pendulum_dynamics import DoublePendulum
from swingup_reference import SwingupReference
from fbl_core.controller import LQR
import fbl_core
import os
import inspect

task = 'training'

# Get the file prefix for the data folder in the learning_fbl package
# On the Cory Lab computers, you may have to enter the absolute path yourself
# e.g.: "/home/cc/ee106b/sp20/staff/ee106b-taa/Desktop/data/ppo_log"

PREFIX = os.path.abspath(os.path.join(os.path.dirname(inspect.getfile(fbl_core)), os.pardir, os.pardir, 'data'))
DIR_NAME = 'pendulum_1'
OUTPUT_DIR = os.path.abspath(os.path.join(PREFIX, DIR_NAME))

try:
    os.mkdir(OUTPUT_DIR)
except OSError as e:
    print(e)
    if task == 'training': # If the task is training error so I don't wipe my training data
        raise 


if task == 'no_learn':
    pass

elif task == 'training':
    import spinup
    import gym
    from gym import spaces
    import learning_fbl_env


    mass1 = 1.0
    mass2 = 1.0
    length1 = 1.0
    length2 = 1.0
    times_step = 0.01
    friction_coeff = 0.5

    true_dynamics = DoublePendulum(mass1, mass2, length1, length2, times_step, friction_coeff)
    nominal_dynamics = DoublePendulum(0.33*mass1, 0.33*mass2, 0.33*length1, 0.33*length2, times_step, friction_coeff)

    A, B, C = nominal_dynamics.linearized_system()
    Q = 200.0 * np.diag([1.0, 0, 1.0, 0])
    R = 1.0 * np.eye(2)

    controller = LQR(A, B, Q, R)
    reference_generator = SwingupReference(nominal_dynamics, 1000)

    action_space = spaces.Box(low=-50, high=50, shape=(6,), dtype=np.float32)
    observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float32)

    fbl_obj = SwingupFBL(action_space = action_space, observation_space = observation_space,
        nominal_dynamics = nominal_dynamics, true_dynamics = true_dynamics,
        controller = controller, reference_generator = reference_generator,
        action_scaling = 1, reward_scaling = 1, reward_norm = 2)

    env = lambda : gym.make('learning_fbl_env:LearningFBLEnv-v0', learn_fbl = fbl_obj)

    spinup.ppo(
        env,
        ac_kwargs={"hidden_sizes":(128,2)},
        seed = np.random.randint(100),
        steps_per_epoch=1250,
        pi_lr=3e-4,
        epochs=2500,
        logger_kwargs = {"output_dir" : OUTPUT_DIR}
        )
    

elif task == 'testing':
    pass




