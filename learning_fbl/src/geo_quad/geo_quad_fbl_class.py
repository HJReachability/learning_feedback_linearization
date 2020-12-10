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

        assert(np.allclose(nominal_dynamics.time_step, true_dynamics.time_step, 1e-8), "nominal and actual dynamics have different timesteps")

        super(TwoControllerQuadFBL, self).__init__(
            action_space, observation_space,
            nominal_dynamics, true_dynamics,
            None, reference_generator,
            action_scaling, reward_scaling, reward_norm)


    def _initial_state_sampler(self):
        """
        Returns the initial state after a reset. This should return an unprocessed state
        """
        # uniformly random xyz coordinates
        # Biased rotation matrix (random velocity vector and integrate)

        return self._reference_generator.sample_state()

    def step(self, a):
        """
        Performs a control action

        a : 
            action (modified parameters)

        """

        x, t = self._get_state_time()

        if self._check_shutdown():
            return None

        ref = self._get_reference()
        # Nominal Control
        v = self._get_v(x, ref)

        # Calculate input
        D, h = self._parse_action(a)
        u = D @ v + h

        # Send control
        self._send_control(u)

        # If needed, wait before reading state again
        self._wait_for_system()

        x_new, t_new = self._get_state_time()

        x_predicted = self._predict_nominal(x, v)

        reward = self._compute_reward(x_predicted, x_new)

        # Check if environment should be reset
        done = self._check_to_reset(x)

        return self._preprocess_state(x), reward, done, {}

    def _get_reference():
        """
        Returns the reference, which should be
            x, b1, v, omega (body), dv, domega
        """

        return super(TwoControllerQuadFBL, self)._get_reference()

    def _get_v(self, y, ref):
        """
        Returns v, the nominal input to the system at linear state y 
        and reference state ref 
        """

        x, R, v, omega = self._nominal_dynamics.split_state(y)
        xd, b1d, vd, omega_d, dv_d, domega_d = self._reference_generator.split_ref(ref)


        ex = x - xd
        ev = v - vd

        v1 = self._controller.pos_controller(ex, ev)

        b3d = v1 - self._nominal_dynamics.mass*(self._nominal_dynamics.gravity*self._nominal_dynamics.e3 + dv_d)
        f = - b3d @ R @ self._nominal_dynamics.e3
        b3d = utils.unitify(b3d) 

        b2d = utils.unitify(np.cross(b3d, utils.unitify(b1d)))
        b1d = utils.unitify(np.cross(b2d, b3d))

        Rd = np.array([b1d, b2d, b3d])

        eR = 0.5*se3.unskew_3d(Rd.transpose() @ R - R.transpose() @ Rd)
        eOmega = omega - R.transpose() @ Rd @ omega_d

        v2 = self._controller.att_controller(eR, eOmega)
        J = self._nominal_dynamics.inertia
        M = v2 - np.cross(omega, J @ omega) - J @ (se3.skew_3d(omega) @ R.transpose() @ Rd @ omega_d - R.transpose() @ Rd @ domega_d)

        v = self._nominal_dynamics.fM_to_u(f, M)

        return v

    def _compute_reward(self, x_predicted, x_actual):
        xp, Rp, vp, wp = self._nominal_dynamics.split_state(x_predicted) # desired
        xa, Ra, va, wa = self._nominal_dynamics.split_state(x_actual) # current

        ex = xa - xp
        ev = va - vp
        eR = 0.5*se3.unskew_3d(Rp.transpose() @ Ra - Ra.transpose() @ Rp)
        ew = wa - Ra.transpose() @ Rp @ wp

        error = np.concatenate([ex, ev, eR, ew])
        return -self._reward_scaling*np.linalg.norm(error, self._reward_norm)


    def _parse_action(self, a):
        """
        Parses the action a, and returns the feedback linearizing terms M and f
        """

        M_flat, f_flat = np.split(self._action_scaling * a, [self._nominal_dynamics.udim**2])

        M = np.reshape(M_flat, (self._nominal_dynamics.udim, self._nominal_dynamics.udim))
        f = np.reshape(f_flat, (self._nominal_dynamics.udim, 1))

        return M, f

    def _predict_nominal(self, x, u):
        """
        Makes a prediction for the nominal model given the input u
        """

        return self._nominal_dynamics.integrate(x, u)

