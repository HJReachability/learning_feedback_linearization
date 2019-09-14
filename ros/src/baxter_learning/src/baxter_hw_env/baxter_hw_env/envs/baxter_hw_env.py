import numpy as np
import gym
import sys
import rospy

from quads_msgs.msg import Transition
from std_msgs.msg import Empty

class BaxterHwEnv(gym.Env):
    def __init__(self):
        # Calling init method of parent class.
        super(BaxterHwEnv, self).__init__()

        # Setting name of ros node.
        self._name = rospy.get_name() + "/Environment"

        # Loading parameters for dynamics
        if not self.load_parameters(): sys.exit(1)
        if not self.register_callbacks(): sys.exit(1)

        # Set up observation space and action space.
        NUM_PREPROCESSED_STATES = 21
        NUM_ACTION_DIMS = 56
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (NUM_PREPROCESSED_STATES,))
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (NUM_ACTION_DIMS,))

        # Queue of state transitions observed in real system with current policy.
        self._transitions = []
        self._num_steps = 0

    def step(self):
        """ Return x, r, u, done. """
        while len(self._transitions) == 0 and not rospy.is_shutdown():
            rospy.logwarn_throttle(10.0, "%s: Out of transitions." % self._name)
            rospy.sleep(0.01)

        if rospy.is_shutdown():
            return None
            
        transition = self._transitions.pop(0)
        x = np.array(transition.x)
        a = np.array(transition.a)
        r = transition.r

        done = False
        if self._num_steps > 25:
            self._num_steps = 0
            done = True

        self._num_steps += 1

        return self.preprocess_state(x), r, a, done, {}

    # def preprocess_state(self, x0):
    #     # x = x0.copy()
    #     # x[0] = np.sin(x[3])
    #     # x[1] = np.sin(x[4])
    #     # x[2]= np.sin(x[5])
    #     # x[3] = np.cos(x[3])
    #     # x[4] = np.cos(x[4])
    #     # x[5]= np.cos(x[5])

    #     # # Remove xi.
    #     # x = np.delete(x, 10)

    #     # TODO: think about removing p, q, r?
    #     return x0

    def preprocess_state(self, x0):
        q = x0[0:7]
        dq = x0[7:14]
        x = np.hstack([np.sin(q), np.cos(q), dq])
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
            self._transition_topic, Transition, self.transition_callback)

        self._linear_system_reset_pub = rospy.Publisher(self._linear_system_reset_topic, Empty)

        return True

    def transition_callback(self, msg):
        self._transitions.append(msg)

    def clear(self):
        self._transitions = []
