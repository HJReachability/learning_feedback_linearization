import gym
import tensorflow as tf
import spinup
import numpy as np

env = lambda : gym.make("quadrotor_14d_env:Quadrotor14dEnv-v0")
# env = lambda : gym.make("Acrobot-v1")

spinup.ppo(env,ac_kwargs={"hidden_sizes":(48,3)},seed = np.random.randint(100),steps_per_epoch=1250, epochs=3000)



