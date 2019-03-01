import torch
import numpy as np

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
        xdot = self._M_q(x) @ u + self._f_q(x)
        return xdot

    def observation(self, x):
        """ Compute y from x. """
        return np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]], [x[3, 0]]])

    def feedback_linearized_output(self, x):
        """ Compute y and its derivatives as needed for feedback linearization. """
        raise NotImplementedError()
        #return np.array([[x[6, 0]], [x[7, 0]], [x[8, 0]], [x[9, 0]]])

    def feedback_linearize(self):
        """
        Computes feedback linearization of this system.
        Returns M matrix and w vector (both functions of state):
                       ``` u = M(x) [ f(x) + v ] ```

        :return: M(x) and f(x) as functions
        :rtype: np.array(np.array), np.array(np.array)
        """
        raise NotImplementedError()

    def _M_q(self, x):
        """ From pp. 34 of linked document above. """
        M_q = np.zeros((self.xdim, self.udim))
        M_q[10, 0] = 1.0
        M_q[11, 1] = 1.0 / self._Ix
        M_q[12, 2] = 1.0 / self._Iy
        M_q[13, 3] = 1.0 / self._Iz
        return M_q

    def _f_q(self, x):
        """ From pp. 33 of linked document above. """
        g17 = -(1.0 / self._mass) * (
            np.sin(x[5, 0]) * np.sin(x[3, 0]) +
            np.cos(x[5, 0]) * np.cos(x[3, 0]) * np.sin(x[4, 0]))
        g18 = -(1.0 / self._mass) * (
            np.cos(x[3, 0]) * np.sin(x[5, 0]) -
            np.cos(x[5, 0]) * np.cos(x[3, 0]) * np.sin(x[4, 0]))
        g19 = -(1.0 / self._mass) * (np.cos(x[5, 0]) * np.cos(x[4, 0]))

        return np.array([
            [x[6, 0]],
            [x[7, 0]],
            [x[8, 0]],
            [x[12, 0] * (np.sin(x[5, 0]) / np.cos(x[4, 0])) +
             x[13, 0] * (np.cos(x[5, 0]) / np.cos(x[4, 0]))],
            [x[12, 0] * np.cos(x[5, 0]) - x[13, 0] * np.sin(x[5, 0])],
            [x[11, 0] + x[12, 0] * (np.sin(x[5, 0]) * np.tan(x[[4, 0]])) +
             x[13, 0] * np.cos(x[5, 0]) * np.tan(x[4, 0])],
            [g17 * x[9, 0]],
            [g18 * x[9, 0]],
            [g19 * x[9, 0]],
            [x[10, 0]],
            [0.0],
            [x[12, 0] * x[13, 0] * (self._Iy - self._Iz) / self._Ix],
            [x[11, 0] * x[13, 0] * (self._Iz - self._Ix) / self._Iy],
            [x[11, 0] * x[12, 0] * (self._Ix - self._Iy) / self._Iz]
        ])
