#!/usr/bin/env python

import numpy as np
from scipy.linalg import expm

from fbl_core.learning_fbl_class import LearningFBL
from pendulum_dynamics import DoublePendulum
import fbl_core.utils as utils

class SwingupFBL(LearningFBL):
    """
    A class for implementing learningFBL controllers and interfacing with the hardware/sim
    This class will be wrapped by an openAI gym environment
    Often, this class will wrap an fbl_core.dynamics class
    """
    def __init__(self, action_space = None, observation_space = None,
        nominal_dynamics = None, true_dynamics = None, reference_type = 'test',
        q = 200.0, r = 1.0,
        action_scaling = 1, reward_scaling = 1, reward_norm = 2):
        """
        Constructor for the superclass. All learningFBL sublcasses should call the superconstructor

        Parameters:
        action_space : :obj:`gym.spaces.Box`
            System action space
        observation_space : :obj:`gym.spaces.Box`
            System state space
        """

        assert(np.all_close(nominal_dynamics.time_step, true_dynamics.time_step, 1e-8), "nominal and actual dynamics have different timesteps")

        super(SwingupFBL, self).__init__(action_space, observation_space)

        self._nominal_dynamics = nominal_dynamics
        self._true_dynamics = true_dynamics
        
        self._action_scaling = action_scaling
        self._reward_scaling = reward_scaling
        self._reward_norm = reward_norm

        self._time_step = self._nominal_dynamics.time_step

        self._x = None
        self._t = None
        self._reset_flag = False

        Q = q*np.diag([1.0, 0, 1.0, 0])
        R = r*np.eye(2)

        A, B, C = self._get_linear_dynamics()
        self._K = utils.solve_LQR(A, B, Q, R)


    def reset():
        """
        All gym environments need a restart method to restart the environment
        This will be called at the start of the learning process and at the end of every epoch
        It must return the (preprocessed) system state
        """

        self._x = self._initial_state_sampler()
        self._t = 0
        self._reference = self._generate_reference_traj(self._x, 1000)
        self._reset_flag = False

        return self._preprocess_state(x)

    def _initial_state_sampler(self):
        lower = np.array([[-np.pi], [-0.5], [-np.pi], [-0.5]])
        upper = -lower
        return np.random.uniform(lower, upper)


    def _get_state_time(self):
        """
        Return the unaugmented system state and the current time
        
        There are no inputs to this function, but you have access to the full system state

        Note: this function should deal with things if the last sensed state is too old
        """

        return self._x, self._t

    def _generate_reference_traj(self, x0, max_path_length):
        """ I ripped this directly from Tyler's code in sandbox and only modified variable names, etc"""


        MAX_CONTINUOUS_TIME_FREQ = 0.1
        MAX_DISCRETE_TIME_FREQ = MAX_CONTINUOUS_TIME_FREQ * self._time_step

        A, B, C = self._nominal_dynamics.linearized_system()

        linsys_xdim=A.shape[0]
        linsys_udim=B.shape[1]

        # Initial y.
        y0 = self._get_linearized_system_state(x0)

        y = np.empty((linsys_xdim, max_path_length))
        for ii in range(linsys_xdim):
            y[ii, :] = np.linspace(
                0, max_path_length * self._time_step,
                max_path_length)
            y[ii, :] = y0[ii, 0] + 1.0 * np.random.uniform() * (1.0 - np.cos(
                2.0 * np.pi * MAX_DISCRETE_TIME_FREQ * \
                np.random.uniform() * y[ii, :])) #+ 0.1 * np.random.normal()

        # Ensure that y ref starts at y0.
        assert(np.allclose(y[:, 0].flatten(), y0.flatten(), 1e-5))

        return np.split(y, indices_or_sections=max_path_length, axis=1)


    def _get_reference(self):
        """
        Return the reference for the system at this timestep

        There are no inputs to this function, but you have access to the full system state
        """

        self._reset_flag = (len(self._reference) <= 1)

        return self._reference[0]

    def _get_linearized_system_state(self, x):
        """
        Returns linear state corresponding to x
        """

        return self._nominal_dynamics.linearized_system_state(x)

    def _get_v(self, y, ref):
        """
        Returns v, the input to the linear system at linear state y 
        and reference state ref 
        (depending on implementation, ref may be a linear state or not)
        """

        diff = self._nominal_dynamics.linear_system_state_delta(ref, y)

        return -np.dot(self._K, diff)

    def _get_Mf(self, x):
        """
        Returns the nominal decoupling matrix M and the nominal drift term f at x
        """

        return self._nominal_dynamics.get_Mf(x)

    def _get_linear_dynamics(self):
        """
        Returns the linear dynamics A, B, C corresponding to the linearized system
        """

        return self._nominal_dynamics.linearized_system()

    def _preprocess_state(self, x):
        """
        Preprocesses the state before sending it to the learning code
        """

        return self._nominal_dynamics.preprocess_state(x)

    def _parse_action(self, a):
        """
        Parses the action a, and returns the feedback linearizing terms M and f
        """

        M_flat, f_flat = np.split(self._action_scaling * a, [4])

        M = np.reshape(M_flat, (self._nominal_dynamics.udim, self._nominal_dynamics.udim))
        f = np.reshape(f_flat, (self._nominal_dynamics.udim, 1))

        return M, f

    def _send_control(self, u):
        """
        Sends the control input to the system
        """

        self._x = self._true_dynamics.integrate(self._x, u)
        self._t = self._t + self._true_dynamics._time_step

        if not self._reset_flag:
            self._reference = self._reference[1:]

        return

    def _wait_for_system(self):
        """
        Waits for system to change after control input is applied

        (if using ROS you'll probably just need rate.sleep() )
        """
        pass
    
    def _compute_reward(self, y_predicted, y_actual):
        """
        Computes the reward for one action
        """
        return -self._reward_scaling * self._nominal_dynamics.observation_distance(y_predicted, y_actual, self._reward_norm)

    def _check_shutdown(self):
        """
        Checks if the code should be shut down (for example if rospy.is_shutdown())
        """
        return False

    def _check_to_reset(self, x):
        """
        If the state goes crazy or something, you can manually reset the environment
        """
        return self._reset_flag




