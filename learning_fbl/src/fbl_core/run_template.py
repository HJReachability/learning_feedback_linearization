#!/usr/bin/env python

import numpy as np
from fbl_core.learning_fbl_class import LearningFBL
from fbl_core.dynamics import Dynamics
import os
import inspect

task = 'no_learn'

# Get the file prefix for the data folder in the learning_fbl package
# On the Cory Lab computers, you may have to enter the absolute path yourself
# e.g.: "/home/cc/ee106b/sp20/staff/ee106b-taa/Desktop/data/ppo_log"

PREFIX = os.path.abspath(os.path.join(os.path.dirname(inspect.getfile(fbl_core)), os.pardir, os.pardir, 'data'))
DIR_NAME = 'pendulum_1'
OUTPUT_DIR = os.path.abspath(os.path.join(PREFIX, DIR_NAME))

try:
    os.mkdir(OUTPUT_DIR)
except OSError as e:
    if task == 'training': # If the task is training error so I don't wipe my training data
        raise 


if task == 'no_learn':
    pass

elif task == 'training':
    import spinup
    import gym

    dynamics = Dynamics()
    fbl_obj = LearningFBL()

    env = lambda : gym.make('LearningFBLEnv-v0', learn_fbl = fbl_obj)

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




