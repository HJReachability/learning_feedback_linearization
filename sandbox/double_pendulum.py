import torch
import numpy as np

from dynamics import Dynamics

class DoublePendulum(Dynamics):
    def __init__(self, mass1, mass2, length1, length2, time_step=0.05):
        self._mass1 = mass1
        self._mass2 = mass2
        self._length1 = length1
        self._length2 = length2
        super(DoublePendulum, self).__init__(4, 2, 2, time_step)

    def __call__(self, x, u):
        """
        Compute xdot from x, u.
        State x is ordered as follows: [theta1, theta1 dot, theta2, theta2 dot].
        """
        xdot = np.zeros(self.xdim)
        xdot[0] = x[1]
        xdot[2] = x[3]

        g = 9.81
        f_q = np.array([
            [-self._mass2 * self._length2 * x[3]**2 * np.sin(
                x[0] - x[2] - g * (self._mass1 + self._mass2))],
            [self._mass2 * self._length1 * x[1]**2 * np.sin(
                x[0] - x[2]) - self._mass2 * g * np.sin(x[2])]
        ])

        off_diag = -self._mass2 * self._length2 * np.cos(x[2] - x[0])
        M_q_inv = (1.0 / (
            (self._mass1 + self._mass2) *
            self._length1 * self._length2 * self._mass2)) * np.array([
                [self._mass2 * self._length2, off_diag],
                [off_diag, (self._mass1 + self._mass2) * self._length2]
            ])

        theta_doubledot = M_q_inv @ (f_q + u)
        xdot[1] = theta_doubledot[0]
        xdot[3] = theta_doubledot[1]

        return xdot

    def observation(self, x):
        """ Compute y from x. """
        pass

    def feedback_linearize(self):
        """
        Computes feedback linearization of this system.
        Returns M matrix and w vector (both functions of state):
                       ``` u = M(x) [ w(x) + v ] ```

        :return: M(x) and w(x) as functions
        :rtype: np.array(np.array), np.array(np.array)
        """
        pass
