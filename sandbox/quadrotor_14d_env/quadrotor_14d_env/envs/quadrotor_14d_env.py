import numpy as np
import gym
import sys
from dynamics import Dynamics
import math

from quadrotor_14d import Quadrotor14D
from scipy.linalg import solve_continuous_are
from gym import spaces


class Quadrotor14dEnv(gym.Env):
    def __init__(self):
        #calling init method of parent class
        super(Quadrotor14dEnv, self).__init__()

         #should we preprocess state
        self._preprocess_state = 1

        if(self._preprocess_state):
            high = np.array([100,100,100,100,100,
            100,100,100,100,100,100,100,100])
        else:
            high = np.array([100,100,100,100,100,
            100,100,100,100,100,100,100,100,100])

        #change in order to change dynamics which quadrotor trains from
        self.action_space = spaces.Box(low=-50,high=50,shape=(20,),dtype=np.float32)
        self.observation_space = spaces.Box(-high,high,dtype=np.float32)
        self._num_steps_per_rollout = 25
        self._reward_scaling = 10
        self._norm = 2

        # TODO: these should match what we get from system identification, once
        # we have reliable numbers there.
        self._mass = 1.0
        Ix = 1.0
        Iy = 1.0
        Iz = 1.0

        self._time_step = 0.01
        self._dynamics = Quadrotor14D(self._mass, Ix, Iy, Iz, self._time_step)

        scaling = 0.33
        self._bad_dynamics = Quadrotor14D(scaling*self._mass, scaling*Ix, scaling*Iy, scaling*Iz, self._time_step)

        self.A,self.B, C=self._dynamics.linearized_system()
        self._count = 0
        self._xdim = self._dynamics.xdim
        self._udim = self._dynamics.udim
        self._M1, self._f1 = self._bad_dynamics.feedback_linearize()
        self._iter_count = 0

    def step(self, u):
        #compute v based on basic control law
        diff = self._dynamics.linear_system_state_delta(self._reference[self._count],self._current_y)

        v = -self._K @ (diff)

        #output of neural network
        m2, f2 = np.split(0.1 * u,[16])

        M = self._bad_dynamics._M_q(self._state) + np.reshape(m2,(self._udim, self._udim))

        f = self._bad_dynamics._f_q(self._state) + np.reshape(f2,(self._udim, 1))

#        print("m1: ", self._bad_dynamics._M_q(self._state))
#        print("m2: ", np.reshape(m2,(self._udim, self._udim)))
#        print("f1: ", self._bad_dynamics._f_q(self._state))
#        print("f2: ", np.reshape(f2,(self._udim, 1)))

        # M = self._bad_dynamics._M_q(self._state)

        # f = self._bad_dynamics._f_q(self._state)

        z = np.dot(M, v) + f

        self._state = self._dynamics.integrate(self._state, z, self._time_step)
        self._current_y = self._dynamics.linearized_system_state(self._state)
#        print("state is: ", self._state)
        reward = self.computeReward(self._y_desired[self._count], self._current_y)
        # reward += self.computePenalty(self._state)

        #Increasing count
        self._count += 1

        #computing observations, rewards, done, info???
        done = False
        if(self._count>=self._num_steps_per_rollout):
            done = True

        list = []
        for x in self._state:
            list.append(x[0])
        observation = list

        if(self._preprocess_state):
            observation = self.preprocess_state(observation)
        #returning stuff
        return np.array(observation), reward, done, {}

    def reset(self):
#        print("reset - new rollout, presumably")

        #gradually increase length of rollouts
        # self._iter_count +=1
        # if(self._iter_count%5000 == 0 and self._num_steps_per_rollout<25):
        # # if(self._iter_count%5000 == 0):
        #     self._num_steps_per_rollout += 1

        #(0) Sample state using state smapler method
        self._state = self.initial_state_sampler()
#        print("initial state is: ", self._state)
        # (1) Generate a time series for v and corresponding y.
        self._reference, self._K = self._generate_reference(self._state)
        self._y_desired = self._generate_ys(self._state,self._reference,self._K)
        self._current_y = self._dynamics.linearized_system_state(self._state)


        #reset internal count
        self._count = 0

        #convert to format algorithm is expecting
        list = []
        for x in self._state:
            list.append(x[0])
        observation = list
        if(self._preprocess_state):
            observation = self.preprocess_state(observation)
        return np.array(observation)

    def seed(self, s):
        np.random.seed(np.random.randomint())


    def render(self):
        # TODO!
        pass


    def initial_state_sampler(self):
        lower0 = np.array([[-0.25, -0.25, -0.25,
                        -0.1, -0.1, -0.1,
                        -0.1, -0.1, -0.1,
                        -1.0, # This is the thrust acceleration - g.
                        -0.1, -0.1, -0.1, -0.1]]).T
        #scaled pitch based on how many episodes to gradually introduce
        lower1 = np.array([[-2.5, -2.5, -2.5,
                        -np.pi / 6.0, -np.pi / 6.0, -np.pi,
                        -0.3, -0.3, -0.3,
                        -3.0, # This is the thrust acceleration - g.
                        -0.3, -0.3, -0.3, -0.3]]).T

        # zero pitch reference
        # lower1 = np.array([[-2.5, -2.5, -2.5,
        #                 -np.pi / 6.0, -np.pi / 6.0, 0,
        #                 -0.3, -0.3, -0.3,
        #                 -3.0, # This is the thrust acceleration - g.
        #                 -0.3, -0.3, -0.3, -0.3]]).T

        frac = 1.0
        lower = frac * lower1 + (1.0 - frac) * lower0
        upper = -lower

        lower[9, 0] = (lower[9, 0] + 9.81) * self._mass
        upper[9, 0] = (upper[9, 0] + 9.81) * self._mass

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

        #random scaling factor for Q based on how many iterations have been done
        Q= 2*self._num_steps_per_rollout * (np.random.uniform() + 0.1) * np.eye(linsys_xdim)
        # Q= 5 * np.eye(linsys_xdim)
        
        #fixed Q scaling
        # Q = 1.0 * np.diag([1.0, 0.0, 0.0, 0.0,
        #                     1.0, 0.0, 0.0, 0.0,
        #                     1.0, 0.0, 0.0, 0.0,
        #                     1.0, 0.0])


        #random R scaling
        #R=1.0 * (np.random.uniform() + 0.1) * np.eye(linsys_udim)

        #fixed R scaling
        R = 1.0 * np.eye(linsys_udim)

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
        y=self._dynamics.linearized_system_state(x)
        ys = []
        for r in refs:
            diff = self._dynamics.linear_system_state_delta(r, y)
            v = -K @ diff
            u = self._dynamics.feedback(x, v)
            x = self._dynamics.integrate(x, u)
            y=self._dynamics.linearized_system_state(x)
            ys.append(y.copy())
        return ys

    def computeReward(self, y_desired, y):
#        print("y desired: ", y_desired)
#        print("y", y)
        return -self._reward_scaling * self._dynamics.observation_distance(
            y_desired, y, self._norm)

    #semi quadratic penalty
    def computePenalty(self,x):
        penalty = 0
        if(x[10]<3.0 or x[10]>16.0):
            penalty -= 3
        if(abs(x[4])>math.pi/3):
            penalty -= 3*(abs(x[4][0])**2)
        if(abs(x[5])>math.pi/3):
            penalty -= 3*(abs(x[5][0])**2)
        return penalty



    def close(self):
        pass

    def preprocess_state(self, x):
        x[0] = np.sin(x[3])
        x[1] = np.sin(x[4])
        x[2]= np.sin(x[5])
        x[3] = np.cos(x[3])
        x[4] = np.cos(x[4])
        x[5]= np.cos(x[5])
<<<<<<< HEAD
        x.remove(10)
        return x

=======
        x.pop(10)
        return x
>>>>>>> 60d5a363556bdc71de77ba66686c67f09b86a189



