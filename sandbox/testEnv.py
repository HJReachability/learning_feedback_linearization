import gym
import tensorflow as tf
import spinup
import numpy as np

env = lambda : gym.make("quadrotor_14d_env:Quadrotor14dEnv-v0")


''' output_dir: algorithm-rewardscaling-dynamicsScaling

    0 dynamics scaling indicates from scratch'''

#polynomial version
# spinup.vpg(
#     env,
#     ac_kwargs={"order" : 1},
#     seed = np.random.randint(100),
#     steps_per_epoch=1250,
#     epochs=2500,
#     logger_kwargs = {"output_dir" : "logs/vpg-modified"}
# )

#mlp version
spinup.ppo(
    env,
    ac_kwargs={"hidden_sizes":(64,2)},
    seed = np.random.randint(100),
    steps_per_epoch=1250,
    epochs=2500,
    logger_kwargs = {"output_dir" : "logs/ppo-10-0.33-5000-v2-preprocess"}
)
