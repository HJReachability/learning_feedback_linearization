import gym
import tensorflow as tf
import spinup
import numpy as np

env = lambda : gym.make("quadrotor_14d_env:Quadrotor14dEnv-v0")


''' output_dir: algorithm-rewardscaling-dynamicsScaling

    0 dynamics scaling indicates from scratch'''

#polynomial version
spinup.vpg(
    env,
    ac_kwargs={"order" : 3},
    seed = np.random.randint(100),
    steps_per_epoch=1250,
    epochs=2500,
    logger_kwargs = {"output_dir" : "logs/poly-10-0.33-null-v2-preprocess-largerQ-start25-uscaling0.1-lr2e-5"}
)

#mlp version

# spinup.ppo(
#     env,
#     ac_kwargs={"hidden_sizes":(64,2)},
#     seed = np.random.randint(100),
#     steps_per_epoch=1250,
#     epochs=2500,
#     logger_kwargs = {"output_dir" : "logs/ppo-10-0.8-null-v2-preprocess-largerQ-start25-uscaling0.1"}
# )
