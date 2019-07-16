import numpy as np
import gym
import sys
from quadrotor_14d import Quadrotor14d
from scipy.linalg import solve_continuous_are

class Quadrotor14dEnv(gym.Env):
    def __init__(self):
        #calling init method of parent class
        super(Quadrotor14dEnv, self).__init__()

        #change in order to change dynamics which quadrotor trains from
        self._num_steps_per_rollout = 25
        self._reward_scaling = 10.0
        self._norm = 2
        self._mass = 1.0
        Ix = 1.0
        Iy = 1.0
        Iz = 1.0
        self._time_step = 0.01
        self._dynamics = Quadrotor14d(self._mass, Ix, Iy, Iz, self._time_step)
        self._count = 0

    def step(self, u):
        #update stuff
        self._dynamics.integrate(self._state, u, self._time_step)
        self._count += 1

        #returns observations, rewards, done, info???
        observation = self._dynamics.linearized_system_state(self._state)
        reward = self.computeReward(self._y_desired[self._count],observation)
        done = False
        if(self._count>=self._num_steps_per_rollout):
            done = True
        return observation, reward, done

    def reset(self):
        #(0) Sample state using state smapler method
        self._state = self.initial_state_sampler()

        # (1) Generate a time series for v and corresponding y.
        reference,K = self._generate_reference(self._state)
        self._y_desired, _ = self._generate_ys(self._state,reference,K)
        observation = self._dynamics.linearized_system_state(self._state)

        self._count = 0
        return observation, reference

    def seed(self, s):
        np.random.seed(0)

    def render(self):
        # TODO!
        pass

   
    def initial_state_sampler(self):
        lower0 = np.array([[-0.25, -0.25, -0.25,
                        -0.1, -0.1, -0.1,
                        -0.1, -0.1, -0.1,
                        -1.0, # This is the thrust acceleration - g.
                        -0.1, -0.1, -0.1, -0.1]]).T
        lower1 = np.array([[-2.5, -2.5, -2.5,
                        -np.pi, -np.pi / 4.0, -np.pi / 4.0,
                        -0.3, -0.3, -0.3,
                        -3.0, # This is the thrust acceleration - g.
                        -0.3, -0.3, -0.3, -0.3]]).T

        frac = 1.0
        lower = frac * lower1 + (1.0 - frac) * lower0
        upper = -lower

        lower[9, 0] = (lower[9, 0] + 9.81) / self._mass
        upper[9, 0] = (upper[9, 0] + 9.81) / self._mass

        return np.random.uniform(lower, upper)

    def _generate_reference(self, x0):
        """
        Use sinusoid with random frequency, amplitude, and bias:
              ``` vi(k) = a * sin(2 * pi * f * k) + b  ```
        """
        MAX_CONTINUOUS_TIME_FREQ = 0.1
        MAX_DISCRETE_TIME_FREQ = MAX_CONTINUOUS_TIME_FREQ * self._dynamics._time_step

        linsys_xdim=self.A.shape[0]
        linsys_udim=self.B.shape[1]
        Q=10.0 * (np.random.uniform() + 0.1) * np.eye(linsys_xdim)
        R=1.0 * (np.random.uniform() + 0.1) * np.eye(linsys_udim)

        # Initial y.
        y0 = self._dynamics.linearized_system_state(x0)

        y = np.empty((linsys_xdim, self._num_steps_per_rollout))
        for ii in range(linsys_xdim):
            y[ii, :] = np.linspace(
                0, self._num_steps_per_rollout * self._dynamics._time_step,
                self._num_steps_per_rollout)
            y[ii, :] = y0[ii, 0] + 1.0 * np.random.uniform() * (1.0 - np.cos(
                2.0 * np.pi * MAX_DISCRETE_TIME_FREQ * \
                np.random.uniform() * y[ii, :])) #+ 0.1 * np.random.normal()

        # Ensure that y ref starts at y0.
        assert(np.allclose(y[:, 0].flatten(), y0.flatten(), 1e-5))

        P = solve_continuous_are(self.A, self.B, Q, R)
        K = np.linalg.inv(R) @ self.B.T @ P
        return (np.split(y, indices_or_sections=self._num_steps_per_rollout, axis=1),K)

    def _generate_ys(self, x0, refs,K):
        """
        Compute desired output sequence given initial state and input sequence.
        This is computed by applying the true dynamics' feedback linearization.
        """
        x = x0.copy()
        ys = []
        xs = []
        for r in refs:
            y=self._dynamics.linearized_system_state(x)
            ys.append(y.copy())
            xs.append(x.copy())
            diff = self._dynamics.linear_system_state_delta(r, y)
            v = -K @ diff
            u = self._dynamics.feedback(x, v)
            x = self._dynamics.integrate(x, u)

        return ys, xs

    def computeReward(self, y_desired, y):
        return -self._reward_scaling * self._dynamics.observation_distance(
            y_desired, y, self._norm)

    def close(self):
        pass
