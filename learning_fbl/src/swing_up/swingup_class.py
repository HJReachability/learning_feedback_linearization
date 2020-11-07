#!/usr/bin/env python

import numpy as np
from scipy.linalg import expm

from fbl_core.dynamics_fbl_class import DynamicsFBL
from pendulum_dynamics import DoublePendulum
import fbl_core.utils as utils

class SwingupFBL(DynamicsFBL):
    """
    A class for implementing learningFBL controllers and interfacing with the hardware/sim
    This class will be wrapped by an openAI gym environment
    Often, this class will wrap an fbl_core.dynamics class
    """
    def __init__(self, action_space = None, observation_space = None,
        nominal_dynamics = None, true_dynamics = None,
        controller = None, reference_generator = None,
        action_scaling = 1, reward_scaling = 1, reward_norm = 2):
        """
        Constructor for the superclass. All learningFBL sublcasses should call the superconstructor

        Parameters:
        action_space : :obj:`gym.spaces.Box`
            System action space
        observation_space : :obj:`gym.spaces.Box`
            System state space
        """

        super(SwingupFBL, self).__init__(action_space, observation_space,
                                        nominal_dynamics, true_dynamics,
                                        controller, reference_generator,
                                        action_scaling, reward_scaling, reward_norm)


    def _initial_state_sampler(self):
        lower = np.array([[-np.pi], [-0.5], [-np.pi], [-0.5]])
        upper = -lower
        return np.random.uniform(lower, upper)


