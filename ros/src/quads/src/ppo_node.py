#!/usr/bin/python

import spinup2
from quadrotor_14d_hw_env.envs.quadrotor_14d_hw_env import Quadrotor14dHwEnv

import rospy
import sys
import numpy as np
import gym
from std_msgs.msg import Empty

def run(msg):
    env = lambda : gym.make("Quadrotor14dHwEnv-v0")
    spinup2.ppo(
        env,
        ac_kwargs={"hidden_sizes":(64,2)},
        seed = np.random.randint(100),
        steps_per_epoch=1250,
        pi_lr=3e-4,
        epochs=2500,
        logger_kwargs = {"output_dir" : "./logs/ppo-log-dir-betterwork"}
    )
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
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
>>>>>>> 47e71d6c3b68b4fbabfe6b9bce313f284bd1ed95


if __name__ == "__main__":
    rospy.init_node("ppo")

    sub = rospy.Subscriber("/in_flight", Empty, run)

    rospy.spin()
>>>>>>> 2f3031e323d5fea502f4d2b06156fa58d5edcffd
