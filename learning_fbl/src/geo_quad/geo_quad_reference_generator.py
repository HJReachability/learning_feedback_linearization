#!/usr/bin/env python

import numpy as np
import se3_utils as se3
import fbl_core.utils as utils
from fbl_core.reference_generator import ReferenceGenerator

class QuadTrajectory(ReferenceGenerator):
    def __init__(self, upper_lims, lower_lims, nominal_dynamics, nominal_traj_length = 20):
        """
        Upper and lower lims are nominal observation state limits    
        """

        self.high_lims = upper_lims
        self.low_lims = lower_lims

        self._nominal_dynamics = nominal_dynamics
        self.nominal_traj_length = nominal_traj_length

        # self.default_high_lims = np.array([10,10,10,1,1,1,1,1,1,1,1,1,2,2,2,1,1,1])
        # self.default_low_lims = - self.default_high_lims

        # self.init_high_lims = np.minimum(self.high_lims, self.default_high_lims)
        # self.init_low_lims = np.maximum(self.low_lims, self.default_low_lims)

    def sample_initial_state(self):
        return sample_state(self)

    def sample_state(self):
        """
        Returns the initial state after a reset. This should return an unprocessed state
        """
        # uniformly random xyz coordinates
        # Biased rotation matrix (random velocity vector and integrate)

        x = np.random.uniform(self.low_lims, self.high_lims)
        p, R, v, omega = self._nominal_dynamics.split_state(x)

        w = utils.unitify(np.random.uniform(-1, 1, (3,)))
        theta = np.random.uniform(-np.pi, np.pi)
        R = se3.rotation_3d(w, theta)

        x = self._nominal_dynamics.unsplit_state(p, R, v, omega)

        return x

    def split_ref(ref):
        """
        splits ref into component parts:
            xd, b1d, vd, omegad, dvd, domegad
        """

        x = ref[:3]
        b1 = ref[3:6]
        v = ref[6:9]
        omega = ref[9:12]
        dv = ref[12:15]
        domega = ref[15:18]

        return x, b1, v, omega, dv, domega


class ConstantTwistQuadTrajectory(QuadTrajectory):
    """docstring for ConstantTwistQuadTrajectory"""
    def __init__(self, upper_lim, lower_lim, nominal_dynamics, nominal_traj_length = 20):
        super(ConstantTwistQuadTrajectory, self).__init__(upper_lim, low_lim, nominal_dynamics, nominal_traj_length)

    def __call__(self, x):
        """ 
        Return a reference trajectory which starts from x. 
        Since the trajectory doesn't really matter, just do a constant velocity geodisic path between the start and goal
        """

        goal_x = self.sample_state()

        ps, Rs, *_ = self._nominal_dynamics.split_state(x) # start
        pg, Rg, *_ = self._nominal_dynamics.split_state(goal_x) # goal

        gs = se3.pR_to_g(ps,Rs)
        gg = se3.pR_to_g(pg,Rg)
        gsg = se3.rbt_inv(gs) @ gg

        xi_diff = se3.inverse_homog_3d(gsg)
        hlim = self.high_lims[12:]
        llim = self.low_lims[12:]

        l = self.nominal_traj_length
        while np.any(xi_diff/(l*self._nominal_dynamics.dt) > hlim) or np.any(xi_diff/(l*self._nominal_dynamics.dt) < llim):
            l = 2*l

        ref = []

        for i in range(l):
            g_i = gs @ se3.homog_3d_exp(xi_diff * self._nominal_dynamics*dt * i)
            p_i, R_i = se3.g_to_pR(g_i)

            b1_i = R_i[0,:] # first column

            p_i = np.reshape(p_i, (3,))
            b1_i = np.reshape(b1_i, (3,))

            ref_i = np.concatenate([p_i, b1_i, xi_diff, np.zeros(6,)])

            ref.append(ref_i)

        return ref

class McClamrochCorkskrew(QuadTrajectory):
    def __init__(self, upper_lim, lower_lim, nominal_dynamics):
        super(McClamrochCorkskrew, self).__init__(upper_lim, lower_lim, nominal_dynamics, nominal_traj_length = np.floor(10/nominal_dynamics.dt))

    def sample_initial_state(self):
        p = np.array([0,0,0])
        v = np.array([0,0,0])
        R = np.eye(3)
        omega = np.array([0,0,0])

        x = self._nominal_dynamics.unsplit_state(p, R, v, omega)

        return x

    def __call__(self, x):

        ref = []
        for i in range(self.nominal_traj_length):
            t = i*self._nominal_dynamics.dt
            pd = np.array([0.4*t, 0.4*np.sin(np.pi*t), 0.6*np.cos(np.pi*t)])
            b1d = np.array([np.cos(np.pi*t), np.sin(np.pi*t), 0])
            ref.append(np.concatenate([pd, b1d, np.zeros((12,))]))

        return ref

        