import numpy as np
import gym
import sys
import math

from scipy.linalg import solve_continuous_are
from gym import spaces


class BallAndBeamEnv(gym.Env):
    def __init__(self,
                 stepsPerRollout,
                 rewardScaling,
                 uscaling):
        # Calling init method of parent class.
        super(BallAndBeamEnv, self).__init__()

        # Setting local parameters.
        self._num_steps_per_rollout = stepsPerRollout
        self._reward_scaling = rewardScaling
        self._norm = 2
        self._time_step = 0.01
        self._uscaling = uscaling

        # Setting action space dimensions so agent knows output size.
        # NOTE: `action` dimension is the number of neural net / learned
        # function outputs.
        NUM_ACTION_DIMS = 1
        self.action_space = spaces.Box(
            low=-50, high=50, shape=(NUM_ACTION_DIMS,), dtype=np.float32)

        #setting observation space dimensions so agent knows input size
        NUM_PREPROCESSED_STATES = 4
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(NUM_PREPROCESSED_STATES,),
            dtype=np.float32)

        self._count = 0
        self._xdim = 4
        self._udim = 1

    def lyapunov(self, x):
        # HACK! Just tryin it out.
        return np.linalg.norm(x)**2

    def lyapunov_dissipation(self, x):
        # HACK! Exponential stability.
        return 0.1 * lyapunov

    def xdot(self, u):
        x1 = self._state[0]
        x2 = self._state[1]
        x3 = self._state[2]
        x4 = self._state[3]

        g = 9.81
        return np.array([
            x2,
            x1 * x4 * x4 - g * np.sin(x3),
            x4,
            u
        ])

    def step(self, a):
        self._count += 1

        # Scale output of neural network.
        scaled_a = self._uscaling * a

        # Integrate dynamics forward with an Euler step.
        self._last_u = self._control(a)
        self._last_state = np.copy(self._state)
        self._state += self._time_step * self.xdot(u)
        reward = self.compute_reward()

        # Check if we're done yet.
        done = self._count >= self._num_steps_per_rollout

        # Formatting observation.
        obs = []
        for x in self._state:
            obs.append(x[0])

        obs = self.preprocess_state(obs)

        return np.array(obs), reward, done, {}

    def control(self, a):
        return -np.linalg.norm(self._state) + a

    def reset(self):
        self._count = 0

        # Sample state using state smapler method
        self._state = self.initial_state_sampler()

        # Formatting observation.
        obs = []
        for x in self._state:
            obs.append(x[0])

        obs = self.preprocess_state(obs)

        return np.array(obs)

    def seed(self, s):
        np.random.seed(np.random.randomint())
        # TODO: figure out how to seed tensorflow...

    def initial_state_sampler(self):
        upper = np.array([1.0, 1.0, 0.3, 0.3])
        lower = -upper

        return np.random.uniform(lower, upper)

    def _generate_reference(self, x0):
        pass

    def compute_reward(self):
        vdot = (self._lyapunov(self._state) -
                self.lyapunov(self._last_state)) / self._time_step
        return -self._reward_scaling * (
            vdot + self.lyapunov_dissipation(self._last_state))

    def preprocess_state(self, x):
        return x
