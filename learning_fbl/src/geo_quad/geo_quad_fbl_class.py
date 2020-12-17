#!/usr/bin/env python

import numpy as np
from scipy.linalg import expm

from fbl_core.dynamics_fbl_class import DynamicsFBL


import fbl_core.utils as utils
import se3_utils as se3

class TwoControllerQuadFBL(DynamicsFBL):
    """
    A parent class for implementing learningFBL controllers for systems described by 
    an fbl_core.dynamics class
    State will be simulated using a "true_dynamics" object, while control will be generated
    using the "nominal_dynamics" object
    Reference trajectories should be provided by a fbl_core.reference_generator class
    Linearized system controller should be provided by a fbl_core.controller class
    """
    def __init__(self, action_space = None, observation_space = None,
        nominal_dynamics = None, true_dynamics = None,
        cascaded_quad_controller = None,
        reference_generator = None,
        action_scaling = 1, reward_scaling = 1, reward_norm = 2):
        """
        Constructor for the superclass. All learningFBL sublcasses should call the superconstructor

        Parameters:
        action_space : :obj:`gym.spaces.Box`
            System action space
        observation_space : :obj:`gym.spaces.Box`
            System state space
        """

        assert np.allclose(nominal_dynamics.time_step, true_dynamics.time_step, 1e-8), "nominal and actual dynamics have different timesteps"

        super(TwoControllerQuadFBL, self).__init__(
            action_space, observation_space,
            nominal_dynamics, true_dynamics,
            cascaded_quad_controller, reference_generator,
            action_scaling, reward_scaling, reward_norm)

        self.upper_bound = self.observation_space.high
        self.lower_bound = self.observation_space.low


    def _initial_state_sampler(self):
        """
        Returns the initial state after a reset. This should return an unprocessed state
        """
        # uniformly random xyz coordinates
        # Biased rotation matrix (random velocity vector and integrate)

        return self._reference_generator.sample_initial_state()

    def step(self, a):
        """
        Performs a control action

        a : 
            action (modified parameters)

        """

        x, t = self._get_state_time()

        # if round(t, 2)%0.05 == 0:
        # print(t)

        if self._check_shutdown():
            return None

        ref = self._get_reference()
        # Nominal Control
        v = self._get_v(x, ref)

        # Calculate input
        D, h = self._parse_action(a)
        u = (np.eye(4) + D) @ v + h

        # if (D > 0.1).any():
        #     print("yo")

        # if utils.checknaninf(u):
        # import pdb
        # pdb.set_trace()

        # Send control
        self._send_control(u)

        # If needed, wait before reading state again
        self._wait_for_system()

        x_new, t_new = self._get_state_time()

        x_predicted = self._predict_nominal(x, v)

        if np.any(x_new > self.upper_bound) or np.any(x_new < self.lower_bound):
            self._reset_flag = True

        if utils.checknaninf(x_new):
            import pdb
            pdb.set_trace()
            if np.isnan(x_new).any():
                print('Found a nan')
            else:
                print('Found an inf')
            self._reset_flag = True

        reward = self._compute_reward(x_predicted, x_new)

        # Check if environment should be reset
        done = self._check_to_reset(x)

        # print('state')
        # print(x)
        # print('ref')
        # print(ref)
        # print('action')
        # print(u)
        # print('x_new')
        # print(x_new)
        # print('x_predicted')
        # print(x_predicted)

        return self._preprocess_state(x), reward, done, {}

    def _get_reference(self):
        """
        Returns the reference, which should be
            rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot
        """

        return super(TwoControllerQuadFBL, self)._get_reference()

    def _get_v(self, y, ref):
        """
        Returns v, the nominal input to the system at linear state y 
        and reference state ref 

        Code based on github.com/fdcl-gwu/uav_geometric_control.git
            - position_control.m
            - attitude_control.m
            - deriv_unit_vector.m
            - test_controller.m
        """

        x, R, v, W = self._nominal_dynamics.split_state(y)
        rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot = self._reference_generator.split_ref(ref)

        # Position and velocity errors
        error_x = x - rx;
        error_v = v - rx_dot;

        # Linear control (force)
        v1 = self._controller.pos_controller(np.concatenate([error_x, error_v]))

        # Desired force vector
        A = v1 - self._nominal_dynamics.mass*self._nominal_dynamics.g*self._nominal_dynamics.e3 \
               + self._nominal_dynamics.mass * rx_2dot

        b3 = R @ self._nominal_dynamics.e3

        # Desired thrust
        f = -np.dot(A, b3);

        ################### Generate the vectors for Rc, Rc_dot, Rc_ddot

        # acceleration error
        error_a = self._nominal_dynamics.g * self._nominal_dynamics.e3 \
                - f / self._nominal_dynamics.mass * b3 \
                - rx_2dot

        # derivative of control input
        v1_dot = self._controller.pos_controller(np.concatenate([error_v, error_a]))
        
        # derivative of desired force vector (desired jerk)
        A_dot = v1_dot + self._nominal_dynamics.mass * rx_3dot

        b3_dot = R @ se3.skew_3d(W) @ self._nominal_dynamics.e3

        # Derivative of desired thrust
        f_dot = - np.dot(A_dot, b3) - np.dot(A, b3_dot)

        # jerk error
        error_j = -f_dot / self._nominal_dynamics.mass * b3 \
                  -f / self._nominal_dynamics.mass * b3_dot \
                  -rx_3dot

        # second derivative of linear control
        v1_ddot = self._controller.pos_controller(np.concatenate([error_a, error_j]))

        # second derivative of desired force vector (desired jounce)
        A_ddot = v1_ddot + self._nominal_dynamics.mass * rx_4dot

        # Desired z axis and derivatives
        b3c, b3c_dot, b3c_ddot = utils.unit_derivatives(-A, -A_dot, -A_ddot)

        # desired pitch axis (we use this intermediate in case rb1 is not a unit vector)
        A2 = -se3.skew_3d(rb1) @ b3c
        A2_dot =  -se3.skew_3d(rb1_dot) @ b3c - se3.skew_3d(rb1) @ b3c_dot
        A2_ddot = -se3.skew_3d(rb1_2dot) @ b3c \
                  - 2 * se3.skew_3d(rb1_dot) @ b3c_dot \
                  - se3.skew_3d(rb1) @ b3c_ddot

        # Desired pitch axis
        b2c, b2c_dot, b2c_ddot = utils.unit_derivatives(A2, A2_dot, A2_ddot)

        # Desired roll axis
        b1c = se3.skew_3d(b2c) @ b3c
        b1c_dot = se3.skew_3d(b2c_dot) @ b3c + se3.skew_3d(b2c) @ b3c_dot
        b1c_ddot = se3.skew_3d(b2c_ddot) @ b3c \
                   + 2 * se3.skew_3d(b2c_dot) @ b3c_dot \
                   + se3.skew_3d(b2c) @ b3c_ddot


        ########### Generate desired R, W, W_dot

        Rc = np.array([b1c, b2c, b3c]).T
        Rc_dot = np.array([b1c_dot, b2c_dot, b3c_dot]).T
        Rc_ddot = np.array([b1c_ddot, b2c_ddot, b3c_ddot]).T

        Wc = se3.unskew_3d(Rc.T @ Rc_dot)
        Wc_dot = se3.unskew_3d(Rc.T @ Rc_ddot - se3.skew_3d(Wc) @ se3.skew_3d(Wc))


        ############ Attitude Control

        error_R = 0.5 * se3.unskew_3d(Rc.T @ R - R.T @ Rc)
        error_W = W - R.T @ Rc @ Wc

        v2 = self._controller.att_controller(np.concatenate([error_R, error_W]))

        J = self._nominal_dynamics.inertia
        M = v2 + se3.skew_3d(R.T @ Rc @ Wc) @ J @ R.T @ Rc @ Wc \
            + J @ R.T @ Rc @ Wc_dot

        v = self._nominal_dynamics.fM_to_u(f, M)

        # ex = x - xd
        # ev = v - vd

        # v1 = self._controller.pos_controller(np.concatenate([ex, ev]))

        # b3d = v1 - self._nominal_dynamics.mass*(self._nominal_dynamics.g*self._nominal_dynamics.e3 + dv_d)
        # f = - b3d @ R @ self._nominal_dynamics.e3
        # b3d = utils.unitify(b3d) 

        # b2d = utils.unitify(np.cross(b3d, utils.unitify(b1d)))
        # b1d = utils.unitify(np.cross(b2d, b3d))

        # Rd = np.array([b1d, b2d, b3d])

        # eR = 0.5*se3.unskew_3d(Rd.transpose() @ R - R.transpose() @ Rd)
        # eOmega = omega - R.transpose() @ Rd @ omega_d

        # v2 = self._controller.att_controller(np.concatenate([eR, eOmega]))
        # J = self._nominal_dynamics.inertia
        # M = v2 - np.cross(omega, J @ omega) - J @ (se3.skew_3d(omega) @ R.transpose() @ Rd @ omega_d - R.transpose() @ Rd @ domega_d)

        # v = self._nominal_dynamics.fM_to_u(f, M)

        # if utils.checknaninf(v):
        # # print(x, ex, ev)
        # # print(R)
        # # print(eR, eOmega, v)

        # import pdb
        # pdb.set_trace()

        return v

    def _compute_reward(self, x_predicted, x_actual):
        xp, Rp, vp, wp = self._nominal_dynamics.split_state(x_predicted) # desired
        xa, Ra, va, wa = self._nominal_dynamics.split_state(x_actual) # current

        ex = xa - xp
        ev = va - vp
        eR = 0.5*se3.unskew_3d(Rp.T @ Ra - Ra.T @ Rp)
        ew = wa - Ra.T @ Rp @ wp

        error = np.concatenate([ex, ev, eR, ew])
        return -self._reward_scaling*np.linalg.norm(error, self._reward_norm)


    def _parse_action(self, a):
        """
        Parses the action a, and returns the feedback linearizing terms M and f
        """

        M_flat, f_flat = np.split(self._action_scaling * a, [self._nominal_dynamics.udim**2])

        M = np.reshape(M_flat, (self._nominal_dynamics.udim, self._nominal_dynamics.udim))
        f = np.reshape(f_flat, (self._nominal_dynamics.udim, ))

        if utils.checknaninf(M):
            import pdb
            pdb.set_trace()
        if utils.checknaninf(f):
            import pdb
            pdb.set_trace()


        return M, f

    def _send_control(self, u):
        """
        Sends the control input to the system
        """

        x = self._true_dynamics.integrate(self._x, u)
        p,R,v,W = self._true_dynamics.split_state(x)
        self._x = self._true_dynamics.unsplit_state(p,R,v,W)
        self._t = self._t + self._true_dynamics.time_step

        if not self._reset_flag:
            self._reference = self._reference[1:]

        if len(self._reference) == 1:
            self._reset_flag = True

        return

    def _predict_nominal(self, x, u):
        """
        Makes a prediction for the nominal model given the input u
        """

        x = self._nominal_dynamics.integrate(x, u)
        p,R,v,W = self._nominal_dynamics.split_state(x)
        return self._nominal_dynamics.unsplit_state(p,R,v,W)


def test_fM_to_u(fbl_obj):

    print('testing fM_to_u and u_to_fM')
    f = 1.7
    M = np.array([1.4,2.1,3.3])

    u = fbl_obj._nominal_dynamics.fM_to_u(f, M)
    f2, M2 = fbl_obj._nominal_dynamics.u_to_fM(u)

    assert np.allclose(f, f2, 1e-8)
    assert np.allclose(M, M2, 1e-8)

    print('Pass')
    return True

def test_get_v(fbl_obj):
    
    print('testing get_v')

    p = np.array([1,2,3])
    w = np.array([1,2,3])
    R = expm(se3.skew_3d(w))
    v = np.array([3.4, 2.3, 1.6])
    W = np.array([2.5, 1.6, 1.3])

    x = fbl_obj._nominal_dynamics.unsplit_state(p,R,v,W)

    
    ref = get_test_ref(1)

    u = fbl_obj._get_v(x, ref)
    f, M = fbl_obj._nominal_dynamics.u_to_fM(u)

    f_desired = 67.5895
    M_desired = np.array([-2.2639, 0.4915, 0.3255])

    print(f)
    print(M)

    assert np.allclose(f, f_desired, 1e-4)
    assert np.allclose(M, M_desired, 1e-4)
    print('Case 1/3 Passed')


    ref = get_test_ref(2)

    u = fbl_obj._get_v(x, ref)
    f, M = fbl_obj._nominal_dynamics.u_to_fM(u)

    f_desired = 52.5378
    M_desired = np.array([-0.7756, 1.1864, 1.2336])

    print(f)
    print(M)

    assert np.allclose(f, f_desired, 1e-4)
    assert np.allclose(M, M_desired, 1e-4)
    print('Case 2/3 Pass')

    p = np.array([0.1, 0.2, 0.3])
    v = np.array([0,0,0])
    R = np.eye(3)
    W = np.array([0,0,0])
    x = fbl_obj._nominal_dynamics.unsplit_state(p,R,v,W)

    ref = np.array([1,2,3,  0,0,0,  0,0,0,  0,0,0,  0,0,0,  1,0,0,  0,0,0,  0,0,0])

    u = fbl_obj._get_v(x, ref)
    f, M = fbl_obj._nominal_dynamics.u_to_fM(u)

    f_desired = -7.38
    M_desired = np.array([-0.6288, -0.3134, -0.2631])

    print(f)
    print(M)

    assert np.allclose(f, f_desired, 1e-4)
    assert np.allclose(M, M_desired, 1e-3)


    print('Case 3/3 Pass')
    return True

def get_test_ref(t):
    rx =      np.array([0.4*t, 0.4  * 1        * np.sin(np.pi*t), 0.6  * 1        * np.cos(np.pi*t)])
    rx_dot =  np.array([  0.4, 0.4  * np.pi    * np.cos(np.pi*t), -0.6 * np.pi    * np.sin(np.pi*t)])
    rx_2dot = np.array([  0.0, -0.4 * np.pi**2 * np.sin(np.pi*t), -0.6 * np.pi**2 * np.cos(np.pi*t)])
    rx_3dot = np.array([  0.0, -0.4 * np.pi**3 * np.cos(np.pi*t), 0.6  * np.pi**3 * np.sin(np.pi*t)])
    rx_4dot = np.array([  0.0, 0.4  * np.pi**4 * np.sin(np.pi*t), 0.6  * np.pi**4 * np.cos(np.pi*t)])
    
    rb1 =      np.array([ 1        * np.cos(np.pi*t), 1         * np.sin(np.pi*t), 0])
    rb1_dot =  np.array([-np.pi    * np.sin(np.pi*t), np.pi     * np.cos(np.pi*t), 0])
    rb1_2dot = np.array([-np.pi**2 * np.cos(np.pi*t), -np.pi**2 * np.sin(np.pi*t), 0])

    ref = np.concatenate([rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot])

    return ref


if __name__ == '__main__':

    from geo_quad_dynamics import GeoQuadDynamics 
    from pos_att_controller import PosAttController 
    from fbl_core.controller import PD 
    from geo_quad_reference_generator import QuadTrajectory
    from gym import spaces

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


    params['pos_p_gain'] = 10
    params['pos_v_gain'] = 8
    params['att_p_gain'] = 1.5
    params['att_v_gain'] = 0.35


    nominal_J = np.diag(params['nominal_J'])
    nominal_dynamics = GeoQuadDynamics(params['nominal_m'], nominal_J, params['nominal_ls'], params['nominal_ctfs'], params['nominal_max_us'],     params['nominal_min_us'], time_step = params['dt'])
    pos_controller = PD(3,3,params['pos_p_gain'], params['pos_v_gain'])
    att_controller = PD(3,3,params['att_p_gain'], params['att_v_gain'])
    controller = PosAttController(pos_controller, att_controller)
    reference_generator = QuadTrajectory(None, None, None, None)

    udim = nominal_dynamics.udim
    xdim = nominal_dynamics.xdim
    action_space = spaces.Box(low=-100, high=100, shape=(udim**2 + udim,), dtype=np.float32)
    observation_space = spaces.Box(low=-100, high=100, shape=(xdim,), dtype=np.float32)

    fbl_obj = TwoControllerQuadFBL(observation_space = observation_space,
                                   action_space = action_space,
                                   nominal_dynamics = nominal_dynamics, 
                                   true_dynamics = nominal_dynamics, 
                                   cascaded_quad_controller = controller,
                                   reference_generator = reference_generator)

    test_fM_to_u(fbl_obj)
    test_get_v(fbl_obj)



