import torch
import numpy as np

from dynamics import Dynamics

class DoublePendulum(Dynamics):
    def __init__(self, mass1, mass2, length1, length2, time_step=0.05):
        self._mass1 = mass1
        self._mass2 = mass2
        self._length1 = length1
        self._length2 = length2

        # Compute mass matrix inverse.
        g = 9.81
        self._M_q_inv = lambda x : (
            1.0 / ((self._mass1 + self._mass2) *
                   self._length1 * self._length2 * self._mass2)) * np.array([
                [self._mass2 * self._length2,
                 -self._mass2 * self._length2 * np.cos(x[2, 0] - x[0, 0])],
                [-self._mass2 * self._length2 * np.cos(x[2, 0] - x[0, 0]),
                (self._mass1 + self._mass2) * self._length2]
            ])

        self._M_q = lambda x : np.linalg.inv(self._M_q_inv(x))

        # Compute coriolis and gravity term.
        self._f_q = lambda x : np.array([
            [-self._mass2 * self._length2 * x[3, 0]**2 * np.sin(
                x[0, 0] - x[2, 0]) - g * (self._mass1 + self._mass2)],
            [self._mass2 * self._length1 * x[1, 0]**2 * np.sin(
                x[0, 0] - x[2, 0]) - self._mass2 * g * np.sin(x[2, 0])]
        ])

        super(DoublePendulum, self).__init__(4, 2, 2, time_step)

    def __call__(self, x, u):
        """
        Compute xdot from x, u.
        State x is ordered as follows: [theta1, theta1 dot, theta2, theta2 dot].
        """
        xdot = np.zeros((self.xdim, 1))
        xdot[0, 0] = x[1, 0]
        xdot[2, 0] = x[3, 0]

        theta_doubledot = self._M_q_inv(x) @ (
            -self._f_q(x) + np.reshape(u, (self.udim, 1)))
        xdot[1, 0] = theta_doubledot[0, 0]
        xdot[3, 0] = theta_doubledot[1, 0]

        return xdot

    def observation(self, x):
        """ Compute y from x. """
        return np.array([[x[0, 0]], [x[2, 0]]])

    def feedback_linearize(self):
        """
        Computes feedback linearization of this system.
        Returns M matrix and w vector (both functions of state):
                       ``` u = M(x) [ w(x) + v ] ```

        :return: M(x) and w(x) as functions
        :rtype: np.array(np.array), np.array(np.array)
        """
        return self._M_q, self._f_q
