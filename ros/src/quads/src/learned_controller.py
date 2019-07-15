import numpy as np
from quadrotor_14d import Quadrotor14D

class LearnedController(object):
    def __init__(self):
        # TODO!
        self._params = []
        self._dynamics = Quadrotor14D(...)
        pass

    def compute_u(self, x, v):
        # Compute u.
        # TODO!
        pass

    def compute_v(self, y_derivs):
        # LQR.
        # TODO!
        pass
