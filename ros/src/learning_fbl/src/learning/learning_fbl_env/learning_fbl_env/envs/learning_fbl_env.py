import numpy as np
import gym
import sys

class LearningFBLEnv(gym.Env):
    """
    A barebones wrapper for the LearningFBL class that works with gym
    """
    def __init__(self, learn_fbl):
        """
        learnfbl should be a LearningFBL object
        """

        # Calling init method of parent class.
        super(LearningFBLEnv, self).__init__()

        self._learn_fbl = learn_fbl

        self.observation_space = self._learn_fbl.observation_space
        self.action_space = self._learn_fbl.action_space

    def step(self, a):
        """ Return x, r, done. """
        return self._learn_fbl.step(a)

    def reset(self):
        return self._learn_fbl.reset()

    def render(self):
        # TODO!
        #aren't we doing this in rviz already?
        pass

    def seed(self, s):
        pass
