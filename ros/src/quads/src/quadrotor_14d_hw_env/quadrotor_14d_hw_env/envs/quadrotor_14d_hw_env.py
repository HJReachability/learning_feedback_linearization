import numpy as np
import gym
import sys
import rospy

from quads_msgs.msg import Transition

class Quadrotor14dHwEnv(gym.Env):
    def __init__(self):
        # Calling init method of parent class.
        super(Quadrotor14dHwEnv, self).__init__()

        # Setting name of ros node.
        self._name = rospy.get_name() + "/Environment"

        # Loading parameters for dynamics
        if not self.load_parameters(): sys.exit(1)
        if not self.register_callbacks(): sys.exit(1)

        # Queue of state transitions observed in real system with current policy.
        self._transitions = []

    def step(self, u):
        """ Return x, v, u, r, done """
        if len(self._transitions) == 0:
            rospy.logerr("%s: Out of transitions.", self._name)
            return None, None, None, None, True

        transition = self._transitions.pop(0)
        x = np.array(transition.x)
#        v = transition.v
#        u = transition.u
        r = transition.r
        return x, r, False, {}

    def reset(self):
        self.clear()

    def render(self):
        # TODO!
        #aren't we doing this in rviz already?
        pass

    def seed(self, s):
        pass

    def load_parameters(self):
        if not rospy.has_param("~topics/transition"):
            return False
        self._transition_topic = rospy.get_param("~topics/transition")

        return True

    def register_callbacks(self):
        self._transition_sub = rospy.Subscriber(
            self._transition_topic, StateTransition, self.transition_callback)

        return True

    def transition_callback(self, msg):
        self._transitions.append(msg)

    def clear(self):
        self._transitions = []
