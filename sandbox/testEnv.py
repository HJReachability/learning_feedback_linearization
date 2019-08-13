import gym
import tensorflow as tf
import spinup2
import numpy as np

#defining arguments for environment
envargs = {"uscaling": 0.1}

#making environment lambda function
env = lambda : gym.make("quadrotor_14d_env:Quadrotor14dEnv-v0", uscaling=0.1)

#vpg
# spinup2.vpg(
#     env,
#     ac_kwargs={"hidden_sizes":(64,2)},
#     seed = np.random.randint(100),
#     steps_per_epoch=1250,
#     epochs=2500,
#     pi_lr=3e-4,
#     logger_kwargs = {"output_dir" : "logs/vpgrandomtest"}
# )

#ppo
spinup2.ppo(
    env,
    ac_kwargs={"hidden_sizes":(64,2)},
    seed = np.random.randint(100),
    steps_per_epoch=1250,
    pi_lr=3e-4,
    epochs=2500,
    logger_kwargs = {"output_dir" : "logs/ppo-randomtest"}
)

#polynomials
# spinup2.vpgpolynomial(
#     env,
#     ac_kwargs={"order":3},
#     seed = np.random.randint(100),
#     steps_per_epoch=1250,
#     epochs=2500,
#     pi_lr=2e-5,
#     l1_scaling=0.001,
#     logger_kwargs = {"output_dir" : "logs/polyrandomtest"}
# )
