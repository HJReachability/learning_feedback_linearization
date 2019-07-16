import numpy as np
import gym
import sys
import rospy

class Quadrotor14dEnv(gym.Env):
    def __init__(self):
        #calling init method of parent class
        super(Quadrotor14dEnv, self). __init__()

        #setting name of ros node????
        self._name = rospy.get_name() + "/Environment"

        # loading parameters for dynamics
        if not self.load_parameters(): sys.exit(1)
        if not self.register_callbacks(): sys.exit(1)

    def step(self, u):
        self._count += 1

        #returns observations, rewards, done, info???
        observation = self._dynamics.linearized_system_state(self._state)
        reward = self.computeReward(self._y_desired[self._count],observation)
        done = False
        if(self._count>=self._num_steps_per_rollout):
            done = True
        return observation, reward, done

    def reset(self):
        # (1) Generate a time series for v and corresponding y.
        reference,K = self._generate_reference(self._state)
    
        observation = self._dynamics.linearized_system_state(self._state)

        self._count = 0
        return observation, reference

    def render(self):
        # TODO!
        #aren't we doing this in rviz already?
        pass
        
    def seed(self, s):
        np.random.seed(0)

    def load_parameters(self):
        if not rospy.has_param("~topics/y"):
            return False
        self._output_derivs_topic = rospy.get_param("~topics/y")

        if not rospy.has_param("~topics/x"):
            return False
        self._state_topic = rospy.get_param("~topics/x")

        if not rospy.has_param("~topics/u"):
            return False
        self._control_topic = rospy.get_param("~topics/u")

        if not rospy.has_param("~dynamics/m"):
            return False
        m = rospy.get_param("~dynamics/m")

        if not rospy.has_param("~dynamics/Ix"):
            return False
        Ix = rospy.get_param("~dynamics/Ix")

        if not rospy.has_param("~dynamics/Iy"):
            return False
        Iy = rospy.get_param("~dynamics/Iy")

        if not rospy.has_param("~dynamics/Iz"):
            return False
        Iz = rospy.get_param("~dynamics/Iz")

        self._mass = m

        self._dynamics = Quadrotor14D(m, Ix, Iy, Iz)

        return True

    def register_callbacks(self):
        self._state_sub = rospy.Subscriber(
            self._state_topic, State, self.state_callback)

        self._output_derivs_sub = rospy.Subscriber(
            self._output_derivs_topic, OutputDerivatives, self.output_callback)

        self._control_pub = rospy.Publisher(self._control_topic, Control)

        return True

    
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

    def computeReward(self, y_desired, y):
        return -self._reward_scaling * self._dynamics.observation_distance(
            y_desired, y, self._norm)
