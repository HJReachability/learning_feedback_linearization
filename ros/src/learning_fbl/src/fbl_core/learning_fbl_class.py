#!/usr/bin/env python

import numpy as np
from scipy.linalg import expm

class LearningFBL(object):
    """
    A class for implementing learningFBL controllers and interfacing with the hardware/sim
    This class will be wrapped by an openAI gym environment
    Often, this class will wrap an fbl_core.dynamics class
    """
    def __init__(self, action_space = None, observation_space = None):
        """
        Constructor for the superclass. All learningFBL sublcasses should call the superconstructor

        Parameters:
        action_space : :obj:`gym.spaces.Box`
            System action space
        observation_space : :obj:`gym.spaces.Box`
            System state space
        """

        # We don't actually need to import gym for this
        self.action_space = action_space
        self.observation_space = observation_space

        # if self._ros_bool:
        #     import rospy
        #     self._setup_ros()
        #     rospy.on_shutdown(self._shutdown_ros)
        #     self._rate = rospy.Rate(rate)

        #     # If there hasn't been a new state measurement in the past epsilon seconds, throw a flag
        #     self._time_epsilon = 1.0/rate 

    def _setup_ros(self):
        """
        Suggested Method (you must call in constructor)

        Sets up the dynamics (sim or hardware), state estimation, and message interface for the problem
    
        Likely you'll want to implement/call the following functions/functionalities:
        - import_packages
        - load_parameters
        - register_callbacks
            - subscribe to system state, do state estimation if needed, then save in a class variable
            - set up control publisher 
            - possibly set up a subscriber to reference, then save in a class variable
        """
        pass


    def _shutdown_ros(self):
        """
        SAFETY METHOD (you must set in constructor):
        Do these things when the code is shut down (ctrl-C)
        """
        pass


    def step(self, a):
        """
        Performs a control action

        a : 
            action (modified parameters)

        """

        x, t = self._get_state_time()

        if self._check_shutdown():
            return None

        y = self._get_linearized_system_state(x)
        ref = self._get_reference()

        # Desired linear input
        v = self._get_v(y, ref)

        # Calculate input
        M1, f1 = self._get_Mf(x)
        M2, f2 = self._parse_action(a)
        u = np.dot(M1 + M2, v) + f1 + f2

        # Send control
        self._send_control(u)

        # If needed, wait before reading state again
        self._wait_for_system()

        x_new, t_new = self._get_state_time()
        y_new = self._get_linearized_system_state(x_new)

        dt = t_new - t

        A, B, C = self._get_linear_dynamics()

        # Since we know nothing about the form of the controller, we convert the linear dynamics to discrete time for prediction
        stack = np.zeros((A.shape(1)+B.shape(1), A.shape(1)+B.shape(1)))
        stack[0:A.shape(0), 0:A.shape(1)] = A
        stack[0:B.shape(0), A.shape(1):] = B
        stack_d = expm(stack * dt)
        A_d = stack[0:A.shape(0), 0:A.shape(1)]
        B_d = stack[0:B.shape(0), A.shape(1):]

        # Predict what the observation at this step should have been
        y_predicted = np.dot(A_d, y) + np.dot(B_d, v)

        reward = self._compute_reward(y_predicted, y_new)

        # Check if environment should be reset
        done = self._check_to_reset(x)

        return self._preprocess_state(x), reward, done, {}

    def reset():
        """
        All gym environments need a restart method to restart the environment
        This will be called at the start of the learning process and at the end of every epoch
        It must return the (preprocessed) system state
        """
        raise NotImplementedError('reset not implemented')
        return self._preprocess_state(x)


    def _get_state_time(self):
        """
        Return the unaugmented system state and the current time
        
        There are no inputs to this function, but you have access to the full system state

        Note: this function should deal with things if the last sensed state is too old
        """
        raise NotImplementedError('get_state_time is not implemented')
        return x, t


    def _get_reference(self):
        """
        Return the reference for the system at this timestep

        There are no inputs to this function, but you have access to the full system state
        """
        raise NotImplementedError('get_reference is not implemented')
        return ref

    def _get_linearized_system_state(self, x):
        """
        Returns linear state corresponding to x
        """
        raise NotImplementedError('_get_linearized_system_state is not implemented')
        return y

    def _get_v(self, y, ref):
        """
        Returns v, the input to the linear system at linear state y 
        and reference state ref 
        (depending on implementation, ref may be a linear state or not)
        """
        raise NotImplementedError('_get_v is not implemented')
        return v

    def _get_Mf(self, x):
        """
        Returns the nominal decoupling matrix M and the nominal drift term f at x
        """

        raise NotImplementedError('_get_Mf is not implemented')
        return M, f

    def _get_linear_dynamics(self):
        """
        Returns the linear dynamics A, B, C corresponding to the linearized system
        """

        raise NotImplementedError('_get_linear_dynamics is not implemented')
        return A, B, C

    def _preprocess_state(self, x):
        """
        Preprocesses the state before sending it to the learning code
        """

        raise NotImplementedError('_preprocess_state is not implemented')
        return x_processed

    def _parse_action(self, a):
        """
        Parses the action a, and returns the feedback linearizing terms M and f
        """
        raise NotImplementedError('parse_action is not implemented')
        return M, f

    def _send_control(self, u):
        """
        Sends the control input to the system
        """
        raise NotImplementedError('send_control is not implemented')
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
        raise NotImplementedError('compute_reward is not implemented')
        return reward

    def _check_shutdown(self):
        """
        Checks if the code should be shut down (for example if rospy.is_shutdown())
        """
        return False

    def _check_to_reset(self, x):
        """
        If the state goes crazy or something, you can manually reset the environment
        """
        return False





