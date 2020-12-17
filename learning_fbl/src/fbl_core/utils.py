#/usr/bin/env python

import numpy as np
from scipy.linalg import solve_continuous_are


def solve_lqr(A,B,Q,R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.dot(np.dot(np.linalg.inv(R), B.T), P)

    return K

def unitify(x):
    """
    make unit vector
    """
    norm = np.linalg.norm(x)
    return x/norm


def unit_derivatives(q, q_dot, q_ddot):
    """
    returns the unit vector and derivatives (u, du, ddu)
    corresponding to the input vector q and its derivatives
    """

    nq = np.linalg.norm(q);
    u = q / nq;

    u_dot = q_dot / nq - q * np.dot(q, q_dot) / nq**3;

    u_ddot = q_ddot / nq - q_dot / nq**3 * (2 * np.dot(q, q_dot)) \
             - q / nq**3 * (np.dot(q_dot, q_dot) + np.dot(q, q_ddot)) \
             + 3 * q / nq**5 * np.dot(q, q_dot)**2

    return (u, u_dot, u_ddot)

def checknaninf(x):
    """
    throws an error if x contains an inf or nan
    """
    if not np.isfinite(x).all():
        # raise ValueError('Contains a Nan or Inf')
        return True
    else:
        return False

