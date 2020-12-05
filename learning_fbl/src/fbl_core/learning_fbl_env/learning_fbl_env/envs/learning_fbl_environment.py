import numpy as np
import gym
from gym import spaces
import sys

class LearningFBLEnv(gym.Env):
    """
    A barebones wrapper for the LearningFBL class that works with gym.
    Automatically normalizes action and observation spaces
    """
    def __init__(self, learn_fbl):
        """
        learnfbl should be a LearningFBL object
        """

        # Calling init method of parent class.
        super(LearningFBLEnv, self).__init__()

        self._learn_fbl = learn_fbl

        action_space = self._learn_fbl.action_space
        if is_instance(action_space, spaces.Box):
            normalize_actions = True
            self.action_low = action_space.low
            self.action_high = action_space.high
            self.action_shift = (action_space.low + action_space.high)/2
            self.action_scale = (action_space.high - action_space.low)/2
            self.action_space = spaces.Box(low=-1, high=1, shape=action_space.shape, dtype = action_space.dtype)
        else:
            normalize_actions = False
            self.action_space = action_space

        observation_space = self._learn_fbl.observation_space
        if is_instance(observation_space, spaces.Box)
            normalize_observations = True
            self.observation_low = observation_space.low
            self.observation_high = observation_space.high
            self.observation_shift = (observation_space.low + observation_space.high)/2
            self.observation_scale = (observation_space.high - observation_space.low)/2
            self.observation_space = spaces.Box(low=-1, high=1, shape=observation_space.shape, dtype=observation_space.dtype)
        else:   
            self.observation_space = observation_space

    def step(self, a):
        """ Return x, r, done. """
        if self.normalize_actions
            a = a*action_scale + action_shift

        x, reward, done, info = self._learn_fbl.step(a)

        if self.normalize_observations
            x = (x - observation_shift)/observation_scale

        return x, reward, done, info

    def reset(self):
        x = self._learn_fbl.reset()

        if self.normalize_observations
            x = (x - observation_shift)/observation_scale

        return x

    def render(self):
        # TODO!
        #aren't we doing this in rviz already?
        pass

    def seed(self, s):
        pass
