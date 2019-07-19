import gym
import tensorflow as tf
import spinup

env = lambda : gym.make("quadrotor_14d_env:Quadrotor14dEnv-v0")

spinup.vpg(env,steps_per_epoch=1250, epochs=4000)




