#!/usr/bin/python

import rospy
import sys
import numpy as np

from quads_msgs.msg import Control
from quads_msgs.msg import State


class SystemIdentifier(object):
    def __init__(self):
        self._state_sub = rospy.Subscriber("/state", State, self.state_callback)
        self._control_sub = rospy.Subscriber("/control/raw", Control, self.control_callback)

        self._states = []
        self._controls = []

    def state_callback(self, msg):
        x = np.array([msg.x, msg.y, msg.z, msg.theta,
                      msg.phi, msg.psi, msg.dx, msg.dy,
                      msg.dz, msg.zeta, msg.xi, msg.q, msg.r, msg.p])

        if len(self._states) <= len(self._controls):
            self._states.append(x)

    def control_callback(self, msg):
        u = np.array([msg.thrustdot2, msg.pitchdot2, msg.rolldot2, msg.yawdot2])

        if len(self._controls) <= len(self._states):
            self._controls.append(u)

    def solve(self):
        num_transitions = len(self._states) - 1
        num_equations = num_transitions * 6

        if num_transitions <= 0:
            return

        A = np.zeros((num_equations, 4))
        b = np.zeros((num_equations, 1))

        sin = np.sin
        cos = np.cos

        for ii in range(num_transitions):
            current_x = self._states[ii]
            current_u = self._controls[ii]
            next_x = self._states[ii + 1]

            theta = current_x[3]
            phi = current_x[4]
            psi = current_x[5]

            # Unknowns are [1/m, 1/Ix, 1/Iy, 1/Iz].
            A[6 * ii, 0] = sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)
            A[6 * ii + 1, 0] = -cos(psi) * sin(phi) + cos(phi) * sin(psi) * sin(theta)
            A[6 * ii + 2, 0] = cos(phi) * cos(theta)
            A[6 * ii + 3, 1] = current_u[1]
            A[6 * ii + 4, 2] = current_u[2]
            A[6 * ii + 5, 3] = current_u[3]

            b[6 * ii, 0] = next_x[6]
            b[6 * ii + 1, 0] = next_x[7]
            b[6 * ii + 2, 0] = next_x[8]
            b[6 * ii + 3, 0] = next_x[11]
            b[6 * ii + 4, 0] = next_x[12]
            b[6 * ii + 5, 0] = next_x[13]

        # Solve A x = b.
        soln, _, _, _ = np.linalg.lstsq(A, b)

        print "Solved!"
        print "-- m = ", soln[0]
        print "-- Ix = ", soln[1]
        print "-- Iy = ", soln[2]
        print "-- Iz = ", soln[3]
