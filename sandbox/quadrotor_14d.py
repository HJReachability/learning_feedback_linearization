import torch
import numpy as np
import math

from dynamics import Dynamics

class Quadrotor14D(Dynamics):
    def __init__(self, mass, Ix, Iy, Iz, time_step=0.05):
        self._mass = mass
        self._Ix = Ix
        self._Iy = Iy
        self._Iz = Iz
        super(Quadrotor14D, self).__init__(14, 4, 4, time_step)

    def __call__(self, x, u):
        """
        Compute xdot from x, u. Please refer to:
        https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf
        for a full derivation of the dynamics. State is laid out as follows:
        ` x = [x, y, z, psi, theta, phi, xdot, ydot, zdot, zeta, xi, p, q, r] `
        ` u = [u1, u2, u3, u4] `
        ` y = [x, y, z, psi] `
        """

        # Drift term. From pp. 33 of linked document above.
        g17 = -(1.0 / self._mass) * (
            np.sin(x[5, 0]) * np.sin(x[3, 0]) +
            np.cos(x[5, 0]) * np.cos(x[3, 0]) * np.sin(x[4, 0]))
        g18 = -(1.0 / self._mass) * (
            np.cos(x[3, 0]) * np.sin(x[5, 0]) -
            np.cos(x[5, 0]) * np.cos(x[3, 0]) * np.sin(x[4, 0]))
        g19 = -(1.0 / self._mass) * (np.cos(x[5, 0]) * np.cos(x[4, 0]))

        drift_term = np.array([
            [x[6, 0]],
            [x[7, 0]],
            [x[8, 0]],
            [x[12, 0] * (np.sin(x[5, 0]) / np.cos(x[4, 0])) +
             x[13, 0] * (np.cos(x[5, 0]) / np.cos(x[4, 0]))],
            [x[12, 0] * np.cos(x[5, 0]) - x[13, 0] * np.sin(x[5, 0])],
            [x[11, 0] + x[12, 0] * (np.sin(x[5, 0]) * math.tan(x[4, 0])) +
             x[13, 0] * np.cos(x[5, 0]) * math.tan(x[4, 0])],
            [g17 * x[9, 0]],
            [g18 * x[9, 0]],
            [g19 * x[9, 0]],
            [x[10, 0]],
            [0.0],
            [x[12, 0] * x[13, 0] * (self._Iy - self._Iz) / self._Ix],
            [x[11, 0] * x[13, 0] * (self._Iz - self._Ix) / self._Iy],
            [x[11, 0] * x[12, 0] * (self._Ix - self._Iy) / self._Iz]
        ])

        # Control coefficient matrix. From pp. 34 of the linked document above.
        control_coefficient_matrix = np.zeros((self.xdim, self.udim))
        control_coefficient_matrix[10, 0] = 1.0
        control_coefficient_matrix[11, 1] = 1.0 / self._Ix
        control_coefficient_matrix[12, 2] = 1.0 / self._Iy
        control_coefficient_matrix[13, 3] = 1.0 / self._Iz

        xdot = control_coefficient_matrix @ u + drift_term
        return xdot

    def observation(self, x):
        """ Compute y from x. """
        return np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]], [x[3, 0]]])

    def feedback_linearize(self):
        """
        Computes feedback linearization of this system.
        Returns M matrix and w vector (both functions of state):
                       ``` u = M(x) [ f(x) + v ] ```

        :return: M(x) and f(x) as functions
        :rtype: np.array(np.array), np.array(np.array)
        """
        return self._M_q, self._f_q

    def _M_q(self, x0):
        """
        v-coefficient matrix in feedback linearization controller. Please refer
        to the file `quad_sym.m` in which we use MATLAB's symbolic toolkit to
        derive this ungodly mess.
        """
        m = self._mass
        Ix = self._Ix
        Iy = self._Iy
        Iz = self._Iz

        # Unpack x.
        x = x0[0, 0]
        y = x0[1, 0]
        z = x0[2, 0]
        psi = x0[3, 0]
        theta = x0[4, 0]
        phi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        p = x0[11, 0]
        q = x0[12, 0]
        r = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        return np.array([
            [-(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))/m, -(zeta*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta)))/(Ix*m), -(zeta*cos(psi)*cos(theta))/(Iy*m), 0],
            [-(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta))/m, -(zeta*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)))/(Ix*m),  (zeta*cos(theta)*sin(psi))/(Iy*m), 0],
            [-(cos(phi)*cos(theta))/m, (zeta*cos(theta)*sin(phi))/(Ix*m), (zeta*sin(theta))/(Iy*m), 0],
            [0, 0, sin(phi)/(Iy*cos(theta)), cos(phi)/(Iz*cos(theta))]
        ])

    def _f_q(self, x0):
        """
        Drift term in feedback linearization controller. Please refer to
        the file `quad_sym.m` in which we use MATLAB's symbolic toolkit to
        derive this ungodly mess.
        """
        m = self._mass
        Ix = self._Ix
        Iy = self._Iy
        Iz = self._Iz

        # Unpack x.
        x = x0[0, 0]
        y = x0[1, 0]
        z = x0[2, 0]
        psi = x0[3, 0]
        theta = x0[4, 0]
        phi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        p = x0[11, 0]
        q = x0[12, 0]
        r = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        # First term.
        f_q1 = [(Ix*Iy*p**2*zeta*sin(phi)*sin(psi) + Ix*Iy*q**2*zeta*sin(phi)*sin(psi) + Ix**2*p*r*zeta*cos(psi)*cos(theta) - Iy**2*q*r*zeta*cos(phi)*sin(psi) - 2*Ix*Iy*q*xi*cos(psi)*cos(theta) - 2*Ix*Iy*p*xi*cos(phi)*sin(psi) + Iy**2*q*r*zeta*cos(psi)*sin(phi)*sin(theta) + Ix*Iy*q*r*zeta*cos(phi)*sin(psi) + Iy*Iz*q*r*zeta*cos(phi)*sin(psi) + 2*Ix*Iy*p*xi*cos(psi)*sin(phi)*sin(theta) + Ix*Iy*p**2*zeta*cos(phi)*cos(psi)*sin(theta) + Ix*Iy*q**2*zeta*cos(phi)*cos(psi)*sin(theta) - Ix*Iy*p*r*zeta*cos(psi)*cos(theta) - Ix*Iz*p*r*zeta*cos(psi)*cos(theta) - Ix*Iy*q*r*zeta*cos(psi)*sin(phi)*sin(theta) - Iy*Iz*q*r*zeta*cos(psi)*sin(phi)*sin(theta))/(Ix*Iy*m)
]

        # Second term.
        f_q2 = [(2*Ix*Iy*q*xi*cos(theta)*sin(psi) - Ix**2*p*r*zeta*cos(theta)*sin(psi) - 2*Ix*Iy*p*xi*cos(phi)*cos(psi) - Iy**2*q*r*zeta*cos(phi)*cos(psi) + Ix*Iy*p**2*zeta*cos(psi)*sin(phi) + Ix*Iy*q**2*zeta*cos(psi)*sin(phi) + Ix*Iy*p*r*zeta*cos(theta)*sin(psi) + Ix*Iz*p*r*zeta*cos(theta)*sin(psi) - Iy**2*q*r*zeta*sin(phi)*sin(psi)*sin(theta) - 2*Ix*Iy*p*xi*sin(phi)*sin(psi)*sin(theta) - Ix*Iy*p**2*zeta*cos(phi)*sin(psi)*sin(theta) - Ix*Iy*q**2*zeta*cos(phi)*sin(psi)*sin(theta) + Ix*Iy*q*r*zeta*cos(phi)*cos(psi) + Iy*Iz*q*r*zeta*cos(phi)*cos(psi) + Ix*Iy*q*r*zeta*sin(phi)*sin(psi)*sin(theta) + Iy*Iz*q*r*zeta*sin(phi)*sin(psi)*sin(theta))/(Ix*Iy*m)
]

        # Third term.
        f_q3 = [(2*Ix*Iy*q*xi*sin(theta) - Ix**2*p*r*zeta*sin(theta) + Iy**2*q*r*zeta*cos(theta)*sin(phi) + Ix*Iy*p*r*zeta*sin(theta) + Ix*Iz*p*r*zeta*sin(theta) + 2*Ix*Iy*p*xi*cos(theta)*sin(phi) + Ix*Iy*p**2*zeta*cos(phi)*cos(theta) + Ix*Iy*q**2*zeta*cos(phi)*cos(theta) - Ix*Iy*q*r*zeta*cos(theta)*sin(phi) - Iy*Iz*q*r*zeta*cos(theta)*sin(phi))/(Ix*Iy*m)
]

        # Fourth term.
        f_q4 = [((q*cos(phi) - r*sin(phi))*(p + r*cos(phi)*tan(theta) + q*sin(phi)*tan(theta)))/cos(theta) + (sin(theta)*(r*cos(phi) + q*sin(phi))*(q*cos(phi) - r*sin(phi)))/cos(theta)**2 + (p*q*cos(phi)*(Ix - Iy))/(Iz*cos(theta)) - (p*r*sin(phi)*(Ix - Iz))/(Iy*cos(theta))
]

        return np.array([f_q1, f_q2, f_q3, f_q4])
