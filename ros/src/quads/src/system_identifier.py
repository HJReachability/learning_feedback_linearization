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
        self._dt = []
        self._last_time = None

    def state_callback(self, msg):
        x = np.array([msg.x, msg.y, msg.z, msg.theta,
                      msg.phi, msg.psi, msg.dx, msg.dy,
                      msg.dz, msg.zeta, msg.xi, msg.q, msg.r, msg.p])

        if len(self._states) <= len(self._controls):
            self._states.append(x)
            t = rospy.Time.now().to_sec()

            if self._last_time is not None:
                self._dt.append(t - self._last_time)

            self._last_time = t

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

        g = 9.81

        for ii in range(num_transitions):
            current_x = self._states[ii]
            current_u = self._controls[ii]
            next_x = self._states[ii + 1]

            theta = current_x[3]
            phi = current_x[4]
            psi = current_x[5]
            zeta = current_x[9]

            # Unknowns are [1/m, 1/Ix, 1/Iy, 1/Iz].
            A[6 * ii, 0] = (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta)) * zeta
            A[6 * ii + 1, 0] = (-cos(psi) * sin(phi) + cos(phi) * sin(psi) * sin(theta)) * zeta
            A[6 * ii + 2, 0] = (cos(phi) * cos(theta)) * zeta
            A[6 * ii + 3, 1] = current_u[1]
            A[6 * ii + 4, 2] = current_u[2]
            A[6 * ii + 5, 3] = current_u[3]

            b[6 * ii, 0] = (next_x[6] - current_x[6]) / self._dt[ii]
            b[6 * ii + 1, 0] = (next_x[7] - current_x[7]) / self._dt[ii]
            b[6 * ii + 2, 0] = (next_x[8] - current_x[8]) / self._dt[ii] + g
            b[6 * ii + 3, 0] = (next_x[11] - current_x[11]) / self._dt[ii]
            b[6 * ii + 4, 0] = (next_x[12] - current_x[12]) / self._dt[ii]
            b[6 * ii + 5, 0] = (next_x[13] - current_x[13]) / self._dt[ii]

        # Solve A x = b.
        soln, _, _, _ = np.linalg.lstsq(A, b)

        print "Solved!"
        print "-- m = ", 1.0 / soln[0]
        print "-- Ix = ", 1.0 / soln[1]
        print "-- Iy = ", 1.0 / soln[2]
        print "-- Iz = ", 1.0 / soln[3]
