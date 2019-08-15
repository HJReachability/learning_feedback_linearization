#!/usr/bin/python

import spinup2

import rospy
import sys


if __name__ == "__main__":
    rospy.init_node("ppo")

    sysid = SystemIdentifier()
    env = lambda : gym.make("quadrotor_14d_hw_env:Quadrotor14dHwEnv-v0")
    spinup2.ppo(
        env,
        ac_kwargs={"hidden_sizes":(64,2)},
        seed = np.random.randint(100),
        steps_per_epoch=1250,
        pi_lr=3e-4,
        epochs=2500,
        logger_kwargs = {"output_dir" : "logs/ppo-randomtest"}
    )
