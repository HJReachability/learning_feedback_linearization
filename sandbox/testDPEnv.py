import gym
import tensorflow as tf
import spinup
import numpy as np

#defining arguments for environment
envargs = {"uscaling": 0.1}

#making environment lambda function
env = lambda : gym.make("quadrotor_14d_env:DoublePendulumEnv-v0", uscaling = 1)

#vpg
# spinup.vpg(
#     env,
#     ac_kwargs={"hidden_sizes":(64,2)},
#     seed = np.random.randint(100),
#     steps_per_epoch=1250,
#     epochs=2500,
#     pi_lr=3e-4,
#     logger_kwargs = {"output_dir" : "logs/vpgrandomtest"}
# )

#ppo
spinup.ppo(
    env,
    ac_kwargs={"hidden_sizes":(64,2)},
    seed = np.random.randint(100),
    steps_per_epoch=1250,
    pi_lr=3e-3,
    epochs=2500,
    logger_kwargs = {"output_dir" : "logs/ppo-dptest-uscaling1-lr3e3"}
)

#polynomials
# spinup.vpgpolynomial(
#     env,
#     ac_kwargs={"order":3},
#     seed = np.random.randint(100),
#     steps_per_epoch=1250,
#     epochs=2500,
#     pi_lr=2e-5,
#     l1_scaling=0.001,
#     logger_kwargs = {"output_dir" : "logs/polyrandomtest"}
# )