#!/usr/bin/env python

import numpy as np

class Controller(object):
    def __init__(self, xdim, udim):
        self.xdim = xdim
        self.udim = udim

    def __call__(self, x):
        """ Compute u from x. """
        raise NotImplementedError()


class LQR(Controller):
    def __init__(self, A, B, Q, R):

        from scipy.linalg import solve_continuous_are

        # try:
        #     assert len(Q.shape) == 2
        # except ValueError as e:
        #     # if Q is a scalar
        #     pass
        # finally:
        #     self.Q = Q

        # try:
        #     assert len(R.shape) == 2
        # except ValueError as e:
        #     # if R is a scalar
        #     pass
        # finally:
        #     self.R = R


        self.Q = Q
        self.R = R

        self.A = A
        self.B = B

        xdim = self.Q.shape[0]
        ydim = self.R.shape[0]

        super(LQR, self).__init__(xdim, ydim)

        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P

    def __call__(self, x):
        """ Compute u from x. """
        return -self.K @ x
