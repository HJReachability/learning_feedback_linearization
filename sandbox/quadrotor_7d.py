import numpy as np

class Quadrotor7D(object):
    def __init__(self, umin, umax):
        """ Construct from np.array bounds on control. """
        # Dimensions.
        self._xdim = 7
        self._udim = 4

        # Control bounds.
        self._umin = umin
        self._umax = umax

        assert len(umin) == self._udim
        assert umin.shape == umax.shape
        assert (umin <= umax).all()

    def xdot(self, x, u):
        assert len(x) == self._xdim
        assert (self._umin <= u).all() and (u <= self._umax).all()

        # State key:          | Control key:
        # x[0] = x position   | u[0] = pitch
        # x[1] = y position   | u[1] = roll
        # x[2] = z position   | u[2] = yaw dot
        # x[3] = vx           | u[3] = thrust (acceleration, not force)
        # x[4] = vy
        # x[5] = vz
        # x[6] = yaw

        g = 9.81

        xdot = np.zeros(self._xdim)
        xdot[0] = x[3]
        xdot[1] = x[4]
        xdot[2] = x[5]
        xdot[3] = u[3] * (np.sin(u[0]) * np.cos(x[6]) +
                          np.sin(u[1]) * np.sin(x[6]))
        xdot[4] = u[3] * (-np.sin(u[1]) * np.cos(x[6]) +
                          np.sin(u[0]) * np.sin(x[6]))
        xdot[5] = u[3] * np.cos(u[1]) * np.cos(u[0]) - g
        xdot[6] = u[2]

        return xdot
