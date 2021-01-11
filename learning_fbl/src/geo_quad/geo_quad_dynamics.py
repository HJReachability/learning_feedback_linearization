#!/usr/bin/env python

import numpy as np
from fbl_core.dynamics import Dynamics
import fbl_core.utils as utils
import se3_utils as se3
import scipy.linalg as spl


class GeoQuadDynamics(Dynamics):
    """
    State is a vector in R^18: x, R(flattened), dx, omega
    Input is a vector in R^4 (prop torques): f1, f2, f3, f4

    #### Not Using ########## Flat variables are x,y,z, e1^T R e1

    """

    def __init__(self, mass, inertia, 
                l, cTau, max_u, min_u,
                g = 9.81, time_step=0.05):
        """

        mass: quad mass
        inertia: 3x3 numpy array; moment of inertia matrix
        l: arm lengths array, len 4
        cTau: prop force to torque ratios, len 4
        max_u: max vals of prop forces
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
        self.max_u = np.array(max_u)
        self.min_u = np.array(min_u)


        self.input_transform = np.array([
                                        [1, 1, 1, 1],
                                        [0, -l[1], 0, l[3]],
                                        [l[0], 0, -l[2], 0],
                                        [-cTau[0], cTau[1], -cTau[2], cTau[3]]
                                        ])

        self.inv_input_transform = np.linalg.inv(self.input_transform)

        self.e1 = np.array([1,0,0])
        self.e2 = np.array([0,1,0])
        self.e3 = np.array([0,0,1])



    def __call__(self, x, u):
        """ Compute xdot from x, u. """

        # import pdb
        # pdb.set_trace()

        u = np.maximum(np.minimum(u, self.max_u), self.min_u)

        f, M = self.u_to_fM(u)

        p, R, v, omega = self.split_state(x)
        omega_hat = se3.skew_3d(omega)

        dp = v
        dR = np.reshape(R @ omega_hat, (9,))
        dv = self.g*self.e3 - f*(R @ self.e3)/self.mass
        domega = self.inv_inertia @ (M - np.cross(omega, self.inertia @ omega))

        dx = np.concatenate([dp, dR, dv, domega])

        # if utils.checknaninf(dx):
        #     import pdb
        #     pdb.set_trace()

        return dx

    def u_to_fM(self, u):
        fM = self.input_transform @ u
        f = fM[0]
        M = fM[1:]
        return f, M

    def fM_to_u(self, f, M):
        fM = np.concatenate([np.array([f]), M])
        u = self.inv_input_transform @ fM

        return u

    def split_state(self, x, renormalize=False):
        """
        Splits the state into p, R, v, omega
        """
        p = x[:3]
        if renormalize:
            R = se3.renormalize_rotation_3d(np.reshape(x[3:12], (3,3)))
        else:
            R = np.reshape(x[3:12], (3,3))
        v = x[12:15]
        omega = x[15:]

        return p, R, v, omega

    def unsplit_state(self, p, R, v, omega):
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

    def wrap_angles(self, x):
        """ We don't have any angles..."""
        return x

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
            dt = 0.25 * self.time_step

        t = 0.0
        x = x0.copy()
        while t < self.time_step - 1e-8:
            # Make sure we don't step past T.

            # while True:
            step = min(dt, self.time_step - t)
            #     dx = step * self.__call__(x, u)
            #     _, R, _, _ = self.split_state(x + 0.5*dx, renormalize=False)
            #     if abs(np.linalg.det(R) - 1) > 0.01:
            #         dt = dt/2
            #     else:
            #         break

            # Use Runge-Kutta order 4 integration. For details please refer to
            # https://en.wikipedia.org/wiki/Runge-Kutta_methods.
            k1 = step * self.__call__(x, u)
            k2 = step * self.__call__(x + 0.5 * k1, u)
            k3 = step * self.__call__(x + 0.5 * k2, u)
            k4 = step * self.__call__(x + k3, u)

            delta_x = (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

            # x = self.discrete_step(x, delta_x)
            x += delta_x

            t += step

        out = self.wrap_angles(x)
        # if utils.checknaninf(out):
        #     import pdb
        #     pdb.set_trace()

        return out

    def discrete_step(self, x, delta_x):

        delta_p, delta_R, delta_v, delta_W = self.split_state(delta_x, renormalize=False)
        p,R,v,W = self.split_state(x, renormalize=False)

        np = p+delta_p
        nv = v+delta_v
        nW = W+delta_W

        w_hat_b = R.T @ delta_R
        nR = R @ spl.expm(w_hat_b)
        # nR = se3.renormalize_rotation_3d(nR)

        return self.unsplit_state(np, nR, nv, nW)









def test_fM_to_u(dynamics):

    print('Testing input input_transform')

    f = 52.5378
    M = np.array([-0.7756, 1.1864, 1.2336])
    u = dynamics.fM_to_u(f, M)

    # print(dynamics.input_transform)
    # print(dynamics.inv_input_transform)
    # print(u)

    u_desired = np.array([-370.2897, 399.6729, -374.0561, 397.2107])

    assert np.allclose(u, u_desired, 1e-4)

    f2, M2 = dynamics.u_to_fM(u)
    assert np.allclose(f, f2, 1e-4)
    assert np.allclose(M, M2, 1e-4)

    print('Pass')
    return True

def test_dynamics(dynamics):

    print('Testing dynamics')

    from scipy.linalg import expm
    p = np.array([1,2,3])
    w = np.array([1,2,3])
    R = expm(se3.skew_3d(w))
    v = np.array([3.4, 2.3, 1.6])
    W = np.array([2.5, 1.6, 1.3])

    x = dynamics.unsplit_state(p,R,v,W)

    f = 52.5378
    M = np.array([-0.7756, 1.1864, 1.2336])
    u = dynamics.fM_to_u(f, M)

    dx = dynamics(x, u)

    dp, dR, dv, dW = dynamics.split_state(dx, renormalize = False)

    dp_desired = np.array([3.4, 2.3, 1.6])
    dR_desired = np.array([[0.7847, -1.8880, 0.2638],
                           [1.1266, 2.5826, -0.0306],
                           [-2.8957, 0.4523, -0.4696]]).T
    dv_desired = np.array([-2.3456, -24.5139, 0.6656])
    dW_desired = np.array([-40.8600, 62.5700, 30.8400])

    assert np.allclose(dp, dp_desired, 1e-4)
    assert np.allclose(dR, dR_desired, 1e-3)
    assert np.allclose(dv, dv_desired, 1e-4)
    assert np.allclose(dW, dW_desired, 1e-4)


    print('Case 1/2 Pass')


    f = 67.5895
    M = np.array([-2.2639, 0.4915, 0.3255])
    u = dynamics.fM_to_u(f, M)

    dx = dynamics(x, u)

    dp, dR, dv, dW = dynamics.split_state(dx, renormalize = False)

    dp_desired = np.array([3.4, 2.3, 1.6])
    dR_desired = np.array([[0.7847, -1.8880, 0.2638],
                           [1.1266, 2.5826, -0.0306],
                           [-2.8957, 0.4523, -0.4696]]).T
    dv_desired = np.array([-3.0176, -31.5370, -1.9542])
    dW_desired = np.array([-115.2750, 27.8250, 8.1375])

    assert np.allclose(dp, dp_desired, 1e-4)
    assert np.allclose(dR, dR_desired, 1e-3)
    assert np.allclose(dv, dv_desired, 1e-4)
    assert np.allclose(dW, dW_desired, 1e-4)


    print('Case 2/2 Pass')
    return True

if __name__ == '__main__':
    params = {}
    params['dt'] = 0.01

    # fdcl_gwo/uav_geometric_control.git matlab   
    params['nominal_m'] = 2
    params['nominal_J'] = [0.02, 0.02, 0.04]
    
    # These don't matter
    params['nominal_ls'] = [0.315, 0.315, 0.315, 0.315]
    params['nominal_ctfs'] = [8.004e-4, 8.004e-4, 8.004e-4, 8.004e-4]
    params['nominal_max_us'] = [1e6, 1e6, 1e6, 1e6]
    params['nominal_min_us'] = [-1e6, -1e6, -1e6, -1e6]

    J = np.diag(params['nominal_J'])
    dynamics = GeoQuadDynamics(params['nominal_m'], J, params['nominal_ls'], params['nominal_ctfs'], params['nominal_max_us'], params['nominal_min_us'], time_step = params['dt'])

    test_fM_to_u(dynamics)
    test_dynamics(dynamics)

