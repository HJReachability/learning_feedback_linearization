import numpy as np
import gym
import sys
import rospy

from quads_msgs.msg import Transition
from std_msgs.msg import Empty

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
        x = np.array([transition.x.x, transition.x.y,
                      transition.x.z, transition.x.theta,
                      transition.x.phi, transition.x.psi,
                      transition.x.dx, transition.x.dy,
                      transition.x.dz, transition.x.zeta,
                      transition.x.xi, transition.x.q,
                      transition.x.r, transition.x.p])
        u = np.array([transition.u.thrustdot2, transition.u.pitchdot2,
                      transition.u.rolldot2, transition.u.yawdot2])
        r = transition.r
        return x, r, u, False, {}

    def preprocess_state(self, x):
        x[0] = np.sin(x[3])
        x[1] = np.sin(x[4])
        x[2]= np.sin(x[5])
        x[3] = np.cos(x[3])
        x[4] = np.cos(x[4])
        x[5]= np.cos(x[5])

        # Remove xi.
        x.pop(10)

        # TODO: think about removing p, q, r?
        return x

    def reset(self):
        self._linear_system_reset_pub.publish(Empty())
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

        if not rospy.has_param("~topics/linear_system_reset"):
            return False
        self._linear_system_reset_topic = rospy.get_param("~topics/linear_system_reset")

        return True

    def register_callbacks(self):
        self._transition_sub = rospy.Subscriber(
            self._transition_topic, StateTransition, self.transition_callback)

        self._linear_system_reset_pub = rospy.Publisher(self._linear_system_reset_topic, Empty)

        return True

    def transition_callback(self, msg):
        self._transitions.append(msg)

    def clear(self):
        self._transitions = []
