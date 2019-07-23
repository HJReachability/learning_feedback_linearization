import gym
import tensorflow as tf
import spinup
import numpy as np

env = lambda : gym.make("quadrotor_14d_env:Quadrotor14dEnv-v0")


''' output_dir: algorithm-rewardscaling-dynamicsScaling

    0 dynamics scaling indicates from scratch'''

spinup.ppo(
    env,
    ac_kwargs={"hidden_sizes":(64,2)},
    seed = np.random.randint(100),
    steps_per_epoch=1250,
    epochs=2500,
    logger_kwargs = {"output_dir" : "logs/sac-take2"}
)
