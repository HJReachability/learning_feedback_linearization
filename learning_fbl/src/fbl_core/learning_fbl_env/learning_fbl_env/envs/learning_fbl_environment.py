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

        self.learn_fbl = learn_fbl

        action_space = self.learn_fbl.action_space
        if isinstance(action_space, spaces.Box):
            self.normalize_actions = True
            self.action_low = action_space.low
            self.action_high = action_space.high
            self.action_shift = (action_space.low + action_space.high)/2
            self.action_scale = (action_space.high - action_space.low)/2
            self.action_space = spaces.Box(low=-1, high=1, shape=action_space.shape, dtype = action_space.dtype)
        else:
            self.normalize_actions = False
            self.action_space = action_space

        observation_space = self.learn_fbl.observation_space
        if isinstance(observation_space, spaces.Box):
            self.normalize_observations = True
            self.observation_low = observation_space.low
            self.observation_high = observation_space.high
            self.observation_shift = (observation_space.low + observation_space.high)/2
            self.observation_scale = (observation_space.high - observation_space.low)/2
            self.observation_space = spaces.Box(low=-1, high=1, shape=observation_space.shape, dtype=observation_space.dtype)
        else: 
            self.normalize_observations = False  
            self.observation_space = observation_space

    def step(self, a):
        """ Return x, r, done. """
        if self.normalize_actions:
            a = self.unnormalize_action(a)

        x, reward, done, info = self.learn_fbl.step(a)

        if self.normalize_observations:
            x = self.normalize_observation(x)

        return x, reward, done, info

    def reset(self):
        x = self.learn_fbl.reset()

        if self.normalize_observations:
            x = self.normalize_observation(x)

        return x

    def render(self):
        # TODO!
        #aren't we doing this in rviz already?
        pass

    def seed(self, s):
        pass

    def normalize_action(self, a):
        a = (a - self.action_shift)/self.action_scale
        return a

    def unnormalize_action(self, a):
        a = a*self.action_scale + self.action_shift
        return a

    def normalize_observation(self, x):
        x = (x - self.observation_shift)/self.observation_scale
        return x

    def unnormalize_observation(self, x):
        x = x*self.observation_scale + self.observation_shift
        return x
