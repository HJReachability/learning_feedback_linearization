import numpy as np
import gym
import sys
from dynamics import Dynamics
import math

from diff_drive import DiffDrive
from scipy.linalg import solve_continuous_are
from gym import spaces


class DiffDriveEnv(gym.Env):
    def __init__(self, stepsPerRollout, rewardScaling, dynamicsScaling, preprocessState, uscaling, largerQ):
        #calling init method of parent class
        super(DiffDriveEnv, self).__init__()

         #setting local parameters
        self._preprocess_state = preprocessState
        self._num_steps_per_rollout = stepsPerRollout
        self._reward_scaling = rewardScaling
        self._norm = 2
        self._time_step = 0.01
        self._uscaling = uscaling
        self._largerQ = largerQ

        #setting action space dimensions so agent knows output size
        self.action_space = spaces.Box(low=-50,high=50,shape=(6,),dtype=np.float32)

        #setting observation space dimensions so agent knows input size
        if(self._preprocess_state):
            self.observation_space = spaces.Box(low=-100,high=100,shape=(5,),dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-100,high=100,shape=(4,),dtype=np.float32)

        # TODO: these should match what we get from system identification, once
        # we have reliable numbers there.
        #setting parameters of quadrotor and creating dynamics object
        self._dynamics = DiffDrive()

        #setting other local variables
        self.A,self.B, C=self._dynamics.linearized_system()
        self._count = 0
        self._xdim = self._dynamics.xdim
        self._udim = self._dynamics.udim
        self._iter_count = 0
        self._M1, self._f1 = self._dynamics.feedback_linearize()
    def step(self, u):
        #compute v based on basic control law
        diff = self._dynamics.linear_system_state_delta(self._reference[self._count],self._current_y)

        v = -self._K @ (diff)

        #output of neural network
        m2, f2 = np.split(self._uscaling*u,[4])

        M = np.reshape(m2,(self._udim, self._udim))# + self._M1(self._state)

        f = np.reshape(f2,(self._udim, 1)) #+ + self._f1(self._state)

        z = np.dot(M, v) + f

        #assert (abs(self._state[3])>1e-3)
        self._state = self._dynamics.integrate(self._state, z, self._time_step)
        self._current_y = self._dynamics.linearized_system_state(self._state)

        reward = self.computeReward(self._y_desired[self._count], self._current_y)

        #Increasing count
        self._count += 1

        #computing observations, rewards, done, info???
        done = False
        if(self._count>=self._num_steps_per_rollout):
            done = True

        #formatting observation
        list = []
        for x in self._state:
            list.append(x[0])
        observation = list

        #preprocessing observation
        if(self._preprocess_state):
            observation = self.preprocess_state(observation)

        #returning stuff
        return np.array(observation), reward, done, {}

    def reset(self):
        #(0) Sample state using state smapler method
        self._state = self.initial_state_sampler()

        # (1) Generate a time series for v and corresponding y.
        self._reference, self._K = self._generate_reference(self._state)
        self._y_desired = self._generate_ys(self._state,self._reference,self._K)
        self._current_y = self._dynamics.linearized_system_state(self._state)

        #reset internal count
        self._count = 0

        #formatting observation
        list = []
        for x in self._state:
            list.append(x[0])
        observation = list

        #preprocessing state
        if(self._preprocess_state):
            observation = self.preprocess_state(observation)

        return np.array(observation)

    def seed(self, s):
        np.random.seed(np.random.randomint())


    def render(self):
        # TODO!
        pass


    def initial_state_sampler(self):
        lower = np.array([[-72], [-72], [-np.pi], [-50]])
        upper = -lower
        return np.random.uniform(lower, upper)

    def _generate_reference(self, x0):
        """
        Use sinusoid with random frequency, amplitude, and bias:
              ``` vi(k) = a * sin(2 * pi * f * k) + b  ```
        """
        MAX_CONTINUOUS_TIME_FREQ = 0.1
        MAX_DISCRETE_TIME_FREQ = MAX_CONTINUOUS_TIME_FREQ * self._dynamics._time_step * 2* np.pi

        linsys_xdim=self.A.shape[0]
        linsys_udim=self.B.shape[1]

        #random scaling factor for Q based on how many iterations have been done
        if(self._largerQ):
            Q= 200 * (np.random.uniform() + 0.1) * np.eye(linsys_xdim)
        else:
            Q= 10 * np.eye(linsys_xdim)

        R = 1.0 * np.eye(linsys_udim)

        # Initial y.
        y0 = self._dynamics.linearized_system_state(x0)

        y = np.empty((linsys_xdim, self._num_steps_per_rollout))
        # for ii in range(linsys_xdim):
        #     y[ii, :] = np.linspace(
        #         0, self._num_steps_per_rollout * self._dynamics._time_step,
        #         self._num_steps_per_rollout)
        #     y[ii, :] = y0[ii, 0] + 1.0 * np.random.uniform() * (1.0 - np.cos(
        #         2.0 * np.pi * MAX_DISCRETE_TIME_FREQ * \
        #         np.random.uniform() * y[ii, :])) #+ 0.1 * np.random.normal()
        y[0, :] = y0[0, 0] + np.random.uniform()*np.sin(MAX_DISCRETE_TIME_FREQ*np.linspace(
                0, self._num_steps_per_rollout * self._dynamics._time_step,
                self._num_steps_per_rollout))
        y[1, :] = y0[1, 0] + np.random.uniform()*np.cos(MAX_DISCRETE_TIME_FREQ*np.linspace(
                0, self._num_steps_per_rollout * self._dynamics._time_step,
                self._num_steps_per_rollout))
        y[2, :] = y0[2, 0] + MAX_DISCRETE_TIME_FREQ*np.random.uniform()*np.cos(MAX_DISCRETE_TIME_FREQ*np.linspace(
                0, self._num_steps_per_rollout * self._dynamics._time_step,
                self._num_steps_per_rollout))
        y[3, :] = y0[3, 0] - MAX_DISCRETE_TIME_FREQ*np.random.uniform()*np.sin(MAX_DISCRETE_TIME_FREQ*np.linspace(
                0, self._num_steps_per_rollout * self._dynamics._time_step,
                self._num_steps_per_rollout))

        # Ensure that y ref starts at y0.
        # assert(np.allclose(y[:, 0].flatten(), y0.flatten(), 1e-5))

        P = solve_continuous_are(self.A, self.B, Q, R)
        K = np.linalg.inv(R) @ self.B.T @ P
        return (np.split(y, indices_or_sections=self._num_steps_per_rollout, axis=1),K)

    def _generate_ys(self, x0, refs,K):
        """
        Compute desired output sequence given initial state and input sequence.
        This is computed by applying the true dynamics' feedback linearization.
        """
        x = x0.copy()
        y=self._dynamics.linearized_system_state(x)
        ys = []
        for r in refs:
            diff = self._dynamics.linear_system_state_delta(r, y)
            v = -K @ diff
            u = self._dynamics.feedback(x, v)
            x = self._dynamics.integrate(x, u)
            y=self._dynamics.linearized_system_state(x)
            # y += (self.A @ y + self.B @ v)*self._time_step
            ys.append(y.copy())
        return ys

    def computeReward(self, y_desired, y):
        return -self._reward_scaling * self._dynamics.observation_distance(y_desired, y, self._norm)

    def close(self):
        pass

    def preprocess_state(self, x):
        return self._dynamics.preprocess_state(x)
