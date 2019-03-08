import torch
import numpy as np

class Dynamics(object):
    def __init__(self, xdim, preprocessed_xdim, udim, ydim, time_step):
        self.xdim = xdim
        self.preprocessed_xdim = preprocessed_xdim
        self.udim = udim
        self.ydim = ydim
        self._time_step = time_step

    def __call__(self, x, u):
        """ Compute xdot from x, u. """
        raise NotImplementedError()

    def observation(self, x):
        """ Compute y from x. """
        raise NotImplementedError()

    def observation_dot(self, x):
        """ Compute y dot from x. """
        raise NotImplementedError()

    def feedback_linearize(self):
        """
        Computes feedback linearization of this system.
        Returns M matrix and w vector (both functions of state):
                       ``` u = M(x) [ w(x) + v ] ```

        :return: M(x) and w(x) as functions
        :rtype: np.array(np.array), np.array(np.array)
        """
        raise NotImplementedError()

    def feedback(self, x, v):
        """ Apply feedback linearization control law to compute u. """
        M_q, f_q = self.feedback_linearize()

        u = M_q(x) @ v + f_q(x)
        return u

    def observation_distance(self, y1, y2,norm):
        """ Compute a distance metric on the observation space. """
        raise NotImplementedError("observation_distance is not implemented.")

    def preprocess_state(self, x):
        """ Preprocess states for input to learned components. """
        raise NotImplementedError("preprocess_state is not implemented.")

    def wrap_angles(self, x):
        """ Wrap angles to [-pi, pi]. """
        raise NotImplementedError("wrap_angles is not implemented.")

    def integrate(self, x0, u, dt=None):
        """
        Integrate initial state x0 (applying constant control u)
        over a time interval of self._time_step, using a time discretization
        of dt.

        :param x0: initial state
        :type x0: np.array
        :param u: control input
        :type u: np.array
        :param dt: time discretization
        :type dt: float
        :return: state after time self._time_step
        :rtype: np.array
        """
        if dt is None:
            dt = 0.25 * self._time_step

        t = 0.0
        x = x0.copy()
        while t < self._time_step - 1e-8:
            # Make sure we don't step past T.
            step = min(dt, self._time_step - t)

            # Use Runge-Kutta order 4 integration. For details please refer to
            # https://en.wikipedia.org/wiki/Runge-Kutta_methods.
            k1 = step * self.__call__(x, u)
            k2 = step * self.__call__(x + 0.5 * k1, u)
            k3 = step * self.__call__(x + 0.5 * k2, u)
            k4 = step * self.__call__(x + k3, u)

            x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += step

        return self.wrap_angles(x)
