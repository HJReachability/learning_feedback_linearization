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
    return x/norm, norm

