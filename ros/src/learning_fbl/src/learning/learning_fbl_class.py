#!/usr/bin/env python

import numpy as np
from scipy.linalg import expm

class LearningFBL(object):
    """
    A class for implementing learningFBL controllers.
    This class will be wrapped by an openAI gym environment
    """
    def __init__(self, ros_bool = False, learning_bool = False, test_bool = False,
            action_space = None, observation_spaces = None, dynamics = None, rate = 20):
        """
        Constructor for the superclass. All learningFBL sublcasses should call the superconstructor

        Parameters:
        ros_bool : bool
            Are you using ROS?
        learning_bool : bool
            Are you doing learning? 
        test_bool : bool
            Are you training or testing?
        action_space : :obj:`gym.spaces.Box`
            System action space
        observation_space : :obj:`gym.spaces.Box`
            System state space
        dynamics : :obj:`learning_general/_dynamics`
            System dynamics object with the following methods:
                - 
        rate : double
            If using ROS, what should the control frequency be?
        """
        self._ros_bool = ros_bool
        self._learning_bool = learning_bool

        self._dynamics = dynamics

        # We don't actually need to import gym for this
        self.action_space = action_space
        self.observation_space = observation_space

        if self._ros_bool:
            import rospy
            self.setup_ros()
            rospy.on_shutdown(self.shutdown_ros)
            self._rate = rospy.Rate(rate)

            # If there hasn't been a new state measurement in the past epsilon seconds, throw a flag
            self._time_epsilon = 1.0/rate 

    def setup_ros(self):
        """

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


    def shutdown_ros(self):
        """
        SAFETY METHOD:
        Do these things when the code is shut down (ctrl-C)
        """
        pass


    def step(self, a):
        """
        Performs an n-step rollout

        a : 
            action

        """

        x, t = self.get_state_time()

        # Cut off if ROS is down and check if input is too old
        if self._ros_bool:
            if rospy.is_shutdown():
                return None
            if rospy.Time.now().to_sec() - t >= self._time_epsilon:
                x, t = self.input_timeout_flag(x, t)


        y = self._dynamics.linearized_system_state(x)
        ref = self.get_reference()

        # Desired linear input
        v = self._dynamics.get_v(y, ref)

        # Calculate input
        M1, f1 = self._dynamics.get_Mf()
        M2, f2 = self.parse_action(a)
        u = np.dot(M1 + M2, v) + f1 + f2

        # Send control
        self.send_control(u)

        # If using ros, wait before reading state again
        if self._ros_bool:
            self._rate.sleep()

        x_new, t_new = self.get_state_time()
        y_new = self._dynamics.linearized_system_state(x_new)

        dt = t_new - t

        A, B, C = self._dynamics.linearized_system()

        # Since we know nothing about the form of the controller, we convert the linear dynamics to discrete time for prediction
        stack = np.zeros((A.shape(1)+B.shape(1), A.shape(1)+B.shape(1)))
        stack[0:A.shape(0), 0:A.shape(1)] = A
        stack[0:B.shape(0), A.shape(1):] = B
        stack_d = expm(stack * dt)
        A_d = stack[0:A.shape(0), 0:A.shape(1)]
        B_d = stack[0:B.shape(0), A.shape(1):]

        # Predict what the observation at this step should have been
        y_predicted = np.dot(A_d, y) + np.dot(B_d, v)

        reward = self.compute_reward(y_predicted, y_new)

        # Check if environment should be reset
        done = self.check_to_reset(x)

        return self._dynamics.preprocess_state(x), reward, done, {}

    def reset():
        """
        All gym environments need a restart method to restart the environment
        This will be called at the start of the learning process and at the end of every epoch
        It must return the (preprocessed) system state
        """
        raise NotImplementedError('reset not implemented')
        return self._dynamics.preprocess_state(x)


    def get_state_time(self):
        """
        Return the unaugmented system state and the current time
        
        There are no inputs to this function, but you have access to the full system state
        """
        raise NotImplementedError('get_state_time is not implemented')
        return x, t


    def get_reference(self):
        """
        Return the reference for the system at this timestep

        There are no inputs to this function, but you have access to the full system state
        """
        raise NotImplementedError('get_reference is not implemented')
        return ref

    def parse_action(self, a):
        """
        Parses the action a, and returns the feedback linearizing terms M and f
        """
        raise NotImplementedError('parse_action is not implemented')
        return M, f

    def send_control(self, u):
        """
        Sends the control input to the system
        """
        raise NotImplementedError('send_control is not implemented')
        return
    
    def compute_reward(self, y_predicted, y_actual):
        """
        Computes the reward for one action
        """
        raise NotImplementedError('compute_reward is not implemented')
        return

    def check_to_reset(self, x):
        """
        If the state goes crazy or something, you can manually reset the environment
        """
        return False

    def input_timeout_flag(self, x, t):
        """
        If the state hasn't updated in the last _time_epsilon seconds, do something
        """
        print "Input is too old. Doing nothing"
        return x, t





