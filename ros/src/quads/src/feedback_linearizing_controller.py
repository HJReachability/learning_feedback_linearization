#import torch
import numpy as np
from scipy.linalg import solve_continuous_are

import sys
import rospy
from quads_msgs.msg import AuxiliaryControl
from quads_msgs.msg import OutputDerivatives
from quads_msgs.msg import Control
from quads_msgs.msg import State

from quadrotor_14d import Quadrotor14D

class FeedbackLinearizingController(object):
    def __init__(self):
        self._name = rospy.get_name() + "/feedback_linearizing_controller"

        if not self.load_parameters(): sys.exit(1)
        if not self.register_callbacks(): sys.exit(1)

        self._M1, self._f1 = self._dynamics.feedback_linearize()
        self._M2 = lambda x : 0.0
        self._f2 = lambda x : 0.0

        # LQR.
        A, B, C = self._dynamics.linearized_system()
        Q = 1.0e-1 * np.diag([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        R = 1.0 * np.eye(4)

        def solve_lqr(A, B, Q, R):
            P = solve_continuous_are(A, B, Q, R)
            K = np.dot(np.dot(np.linalg.inv(R), B.T), P)
            return K


        self._K = solve_lqr(A, B, Q, R)
        self._ref = np.zeros((14, 1))
        self._ref[8, 0] = 1.0

        self._y = None

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

        self._dynamics = Quadrotor14D(m, Ix, Iy, Iz)

        return True

    def register_callbacks(self):
        self._state_sub = rospy.Subscriber(
            self._state_topic, State, self.state_callback)

        self._output_derivs_sub = rospy.Subscriber(
            self._output_derivs_topic, OutputDerivatives, self.output_callback)

        self._control_pub = rospy.Publisher(self._control_topic, Control)

        return True

    def state_callback(self, msg):
        # Update x.
        x = np.array([[msg.x], [msg.y], [msg.z], [msg.theta],
                      [msg.phi], [msg.psi], [msg.dx], [msg.dy],
                      [msg.dz], [msg.zeta], [msg.xi], [msg.q], [msg.r], [msg.p]])

        # Determine v.
        if self._y is not None:
            v = -np.dot(self._K, (self._y - self._ref))

            u = self.feedback(x, v) #.detach().numpy().copy()
            u_msg = Control()
            u_msg.u1 = u[0, 0]
            u_msg.u2 = u[1, 0]
            u_msg.u3 = u[2, 0]
            u_msg.u4 = u[3, 0]
            self._control_pub.publish(u_msg)

    def output_callback(self, msg):
        self._y = np.array([[msg.x], [msg.xdot1], [msg.xdot2], [msg.xdot3],
                            [msg.y], [msg.ydot1], [msg.ydot2], [msg.ydot3],
                            [msg.z], [msg.zdot1], [msg.zdot2], [msg.zdot3],
                            [msg.psi], [msg.psidot1]])

    def feedback(self, x, v):
        """ Compute u from x, v (np.arrays). See above comment for details. """
        v = np.reshape(v, (4, 1))

#        print "M1 v = ", np.dot(self._M1(x), v)
#        print "f1 = ", self._f1(x)

        return np.dot(self._M1(x), v) + self._f1(x)

#        M = torch.from_numpy(self._M1(x)).float() + torch.reshape(
#            self._M2(torch.from_numpy(x.flatten()).float()),
#            (self._udim, self._udim))
#        f = torch.from_numpy(self._f1(x)).float() + torch.reshape(
#            self._f2(torch.from_numpy(x.flatten()).float()),
#            (self._udim, 1))

        # TODO! Make sure this is right (and consistent with dynamics.feedback).
#        return torch.mm(M, torch.from_numpy(v).float()) + f

#    def noisy_feedback(self, x, v):
#        """ Compute noisy version of u given x, v (np.arrays). """
#        return torch.distributions.normal.Normal(
#            self.feedback(x, v),
#            self._noise_scaling * torch.abs(self._noise_std_variable) + 0.1)

#    def sample_noisy_feedback(self, x, v):
#        """ Compute noisy version of u given x, v (np.arrays). """
#        return self.noisy_feedback(x, v).sample().numpy()
