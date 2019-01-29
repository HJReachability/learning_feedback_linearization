import torch
import numpy as np

from dynamics import Dynamics

class DoublePendulum(Dynamics):
    def __init__(self, time_step=0.05):
        super(DoublePendulum, self).__init__(4, 2, 2, time_step)

    def __call__(self, x, u):
        """ Compute xdot from x, u. """
        pass

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
