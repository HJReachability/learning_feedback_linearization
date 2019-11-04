#!/usr/bin/env python

import rospy
import baxter_interface
from baxter_pykdl import baxter_kinematics
import utils
import numpy as np


if __name__ == '__main__':\

    rospy.init_node('pykdl_test')

    limb = baxter_interface.Limb('right')
    kin = baxter_kinematics('right')

    print kin.forward_position_kinematics()
    current_position = utils.get_joint_positions(limb).reshape((7,1))
    print kin.forward_position_kinematics(joint_values=utils.joint_array_to_dict(current_position, limb))

