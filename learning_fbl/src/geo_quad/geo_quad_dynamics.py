#!/usr/bin/env python

import numpy as np
from fbl_core.dynamics import Dynamics
import se3_utils as se3


class GeoQuadDynamics(Dynamics):
    """
    State is a vector in R^18: x, R(flattened), dx, omega
    Input is a vector in R^4 (prop torques): f1, f2, f3, f4

    #### Not Using ########## Flat variables are x,y,z, e1^T R e1

    """

    def __init__(self, mass, inertia, 
                l, cTau,
                g = 9.81, time_step=0.05):
        """

        mass: quad mass
        inertia: 3x3 numpy array; moment of inertia matrix
        l: arm lengths array, len 4
        cTau: prop force to torque ratios, len 4
        """
        
        xdim = 18
        preprocessed_xdim = 18
        udim = 4
        ydim = 18

        super(GeoQuadDynamics, self).__init__(xdim, preprocessed_xdim, udim, ydim, time_step)

        self.mass = mass

        if not inertia.shape == (3,3):
            raise TypeError('quad inertia must be a 3x3 matrix')
        self.inertia = inertia
        self.inv_inertia = np.linalg.inv(inertia)
        self.g = g


        self.input_transform = np.array([
                                        [1, 1, 1, 1]
                                        [0, -l[1], 0, l[3]]
                                        [l[0], 0, -l[2], 0]
                                        [-cTau[0], cTau[1], -cTau[2], cTau[3]]
                                        ])

        self.inv_input_transform = np.linalg.inv(self.input_transform)

        self.e1 = np.array([1,0,0])
        self.e2 = np.array([0,1,0])
        self.e3 = np.array([0,0,1])



    def __call__(self, x, u):
        """ Compute xdot from x, u. """
        f, M = u_to_fM(u)

        p, R, v, omega = split_state(x)
        omega_hat = se3.skew_3d(omega)

        dp = v
        dR = np.reshape(R @ omega_hat, (9,))
        dv = self.mass*self.g*self.e3 + f*(R @ self.e3)
        domega = self.inv_inertia @ (M - np.cross(omega, self.inertia @ omega))

        dx = np.concatenate([dp, dR, dv, domega])
        return dx

    def u_to_fM(u):
        fM = self.input_transform @ u
        f = fM[0]
        M = fM[1:]
        return f, M

    def fM_to_u(f, M):
        fM = np.concatenate([np.array([f]), M])
        u = self.inv_input_transform @ fM
        return u

    def split_state(x):
        """
        Splits the state into p, R, v, omega
        """
        p = x[:3]
        R = se3.renormalize_rotation_3d(np.reshape(x[3:12], (3,3)))
        v = x[12:15]
        omega = x[15:]

        return p, R, v, omega

    def unsplit_state(p, R, v, omega):
        """
        Combines the parts into a state vector
        """
        x = np.zeros((self.xdim, ))
        x[:3] = p
        x[3:12] = np.reshape(R, (9,))
        x[12:15] = v
        x[15:] = omega 

        return x

    def preprocess_state(self, x):
        """ Preprocess states for input to learned components. """
        return x
