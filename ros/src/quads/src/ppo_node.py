#!/usr/bin/python

import spinup2
<<<<<<< HEAD

import rospy
import sys


if __name__ == "__main__":
    rospy.init_node("ppo")

    sysid = SystemIdentifier()
    env = lambda : gym.make("quadrotor_14d_hw_env:Quadrotor14dHwEnv-v0")
=======
from quadrotor_14d_hw_env.envs.quadrotor_14d_hw_env import Quadrotor14dHwEnv

import rospy
import sys
import numpy as np
import gym
from std_msgs.msg import Empty

def run(msg):
    env = lambda : gym.make("Quadrotor14dHwEnv-v0")
>>>>>>> 2f3031e323d5fea502f4d2b06156fa58d5edcffd
    spinup2.ppo(
        env,
        ac_kwargs={"hidden_sizes":(64,2)},
        seed = np.random.randint(100),
        steps_per_epoch=1250,
        pi_lr=3e-4,
        epochs=2500,
        logger_kwargs = {"output_dir" : "logs/ppo-randomtest"}
    )
<<<<<<< HEAD
=======


if __name__ == "__main__":
    rospy.init_node("ppo")

    sub = rospy.Subscriber("/in_flight", Empty, run)

    rospy.spin()
>>>>>>> 2f3031e323d5fea502f4d2b06156fa58d5edcffd
