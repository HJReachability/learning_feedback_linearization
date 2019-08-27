import rospy
import numpy as np

from quads_msgs.msg import OutputDerivatives
import sys

class DataCollector(object):
    def __init__(self):
        self._ref_sub = rospy.Subscriber("/ref/linear_system", OutputDerivatives, self.ref_callback)
        self._linear_system_state_sub = rospy.Subscriber("/output_derivs", OutputDerivatives, self.linear_system_state_callback)

        self._refs = []
        self._ref_times = []
        self._linear_system_states = []
        self._linear_system_state_times = []

    def ref_callback(self, msg):
        t = rospy.Time.now().to_sec()
        x = np.array([msg.x, msg.xdot1, msg.xdot2, msg.xdot3,
                      msg.y, msg.ydot1, msg.ydot2, msg.ydot3,
                      msg.z, msg.zdot1, msg.zdot2, msg.zdot3,
                      msg.psi, msg.psidot1])

        self._refs.append(x)
        self._ref_times.append(t)

    def linear_system_state_callback(self, msg):
        t = rospy.Time.now().to_sec()
        x = np.array([msg.x, msg.xdot1, msg.xdot2, msg.xdot3,
                      msg.y, msg.ydot1, msg.ydot2, msg.ydot3,
                      msg.z, msg.zdot1, msg.zdot2, msg.zdot3,
                      msg.psi, msg.psidot1])

        self._linear_system_states.append(x)
        self._linear_system_state_times.append(t)

    def dump(self):
        """ Dump to disk. """
        import dill
        PREFIX = "/home/hysys/Github/learning_feedback_linearization/ros/src/quads/data/"

        f = open(PREFIX + "refs.pkl", "wb")
        dill.dump(self._refs, f)
        f.close()
        f = open(PREFIX + "ref_times.pkl", "wb")
        dill.dump(self._ref_times, f)
        f.close()
        f = open(PREFIX + "linear_system_states.pkl", "wb")
        dill.dump(self._linear_system_states, f)
        f.close()
        f = open(PREFIX + "linear_system_state_times.pkl", "wb")
        dill.dump(self._linear_system_state_times, f)
        f.close()
