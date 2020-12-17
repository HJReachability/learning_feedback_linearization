#!/usr/bin/env python

import numpy as np
import se3_utils as se3
import fbl_core.utils as utils
from fbl_core.reference_generator import ReferenceGenerator
from scipy.linalg import expm

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
        return self.sample_state()

    def sample_state(self):
        """
        Returns the initial state after a reset. This should return an unprocessed state
        """
        # uniformly random xyz coordinates
        # Biased rotation matrix (random velocity vector and integrate)

        x = np.random.uniform(self.low_lims, self.high_lims)
        p, R, v, omega = self._nominal_dynamics.split_state(x, renormalize=False)

        w = utils.unitify(np.random.uniform(-1, 1, (3,)))
        theta = np.random.uniform(-0.1*np.pi, 0.1*np.pi)
        R = se3.rotation_3d(w, theta)

        x = self._nominal_dynamics.unsplit_state(p, R, v, omega)

        if utils.checknaninf(x):
            import pdb
            pdb.set_trace()

        return x

    def split_ref(self, ref):
        """
        splits ref into component parts:
            rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot
        """

        
        rx = ref[:3] 
        rx_dot = ref[3:6]
        rx_2dot = ref[6:9]
        rx_3dot = ref[9:12]
        rx_4dot = ref[12:15]
        rb1 = ref[15:18]
        rb1_dot = ref[18:21]
        rb1_2dot = ref[21:24]

        return rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot


# class ConstantTwistQuadTrajectory(QuadTrajectory):
#     """docstring for ConstantTwistQuadTrajectory"""
#     def __init__(self, upper_lim, lower_lim, nominal_dynamics, nominal_traj_length = 20):
#         super(ConstantTwistQuadTrajectory, self).__init__(upper_lim, lower_lim, nominal_dynamics, nominal_traj_length)

#     def __call__(self, x):
#         """ 
#         Return a reference trajectory which starts from x. 
#         Since the trajectory doesn't really matter, just do a constant velocity geodisic path between the start and goal
#         """

#         goal_x = self.sample_state()

#         ps, Rs, *_ = self._nominal_dynamics.split_state(x) # start
#         pg, Rg, *_ = self._nominal_dynamics.split_state(goal_x) # goal

#         gs = se3.pR_to_g(ps,Rs)
#         gg = se3.pR_to_g(pg,Rg)
#         gsg = se3.rbt_inv(gs) @ gg

#         xi_diff = se3.inverse_homog_3d(gsg)
#         hlim = self.high_lims[12:]
#         llim = self.low_lims[12:]

#         l = self.nominal_traj_length
#         while np.any(xi_diff/(l*self._nominal_dynamics.dt) > hlim) or np.any(xi_diff/(l*self._nominal_dynamics.dt) < llim):
#             l = 2*l

#         ref = []

#         for i in range(l):
#             g_i = gs @ se3.homog_3d_exp(xi_diff * self._nominal_dynamics.dt * i)
#             p_i, R_i = se3.g_to_pR(g_i)

#             b1_i = R_i[0,:] # first column

#             p_i = np.reshape(p_i, (3,))
#             b1_i = np.reshape(b1_i, (3,))

#             ref_i = np.concatenate([p_i, b1_i, xi_diff, np.zeros(6,)])

#             ref.append(ref_i)

#         b1g = np.reshape(Rg[0,:], (3,))
#         ref_g = np.concatenate([pg, b1g, np.zeros(6,), np.zeros(6,)])
#         ref.append(ref_g)

#         if utils.checknaninf(ref):
#             import pdb
#             pdb.set_trace()

#         return ref

class Lissajous(QuadTrajectory):
    def __init__(self, upper_lim, lower_lim, nominal_dynamics):
        super(Lissajous, self).__init__(upper_lim, lower_lim, nominal_dynamics, nominal_traj_length = int(20//nominal_dynamics.dt))

    def sample_initial_state(self):
        p = np.array([0,0,0])
        v = np.array([0,0,0])
        R = expm(se3.skew_3d(np.pi*np.array([0,0,1])))
        omega = np.array([0,0,0])

        x = self._nominal_dynamics.unsplit_state(p, R, v, omega)

        return x

    def __call__(self, x):

        A = 1
        B = 1
        C = 0.2

        d = 0

        a = 1
        b = 2
        c = 2
        alt = -1

        w = 2*np.pi/10

        ref = []
        for i in range(self.nominal_traj_length):
            t = i*self._nominal_dynamics.dt

            rx =      np.array([A*np.sin(a*t+d), B*np.sin(b*t), alt+C*np.cos(c*t)])
            rx_dot =  np.array([A*a*np.cos(a*t+d), B*b*np.cos(b*t), C*c*-np.sin(c*t)])
            rx_2dot = np.array([A*a**2*-np.sin(a*t+d), B*b**2*-np.sin(b*t), C*c**2*-np.cos(c*t) ])
            rx_3dot = np.array([A*a**3*-np.cos(a*t+d), B*b**3*-np.cos(b*t), C*c**3*np.sin(c*t)])
            rx_4dot = np.array([A*a**4*np.sin(a*t+d), B*b**4*np.sin(b*t), C*c**4*np.cos(c*t)])
            
            rb1 =      np.array([np.cos(w*t), np.sin(w*t), 0])
            rb1_dot =  np.array([w*-np.sin(w*t), w*np.cos(w*t), 0])
            rb1_2dot = np.array([w**2*-np.cos(w*t), w**2*-np.sin(w*t), 0])

            ref.append(np.concatenate([rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot]))

        if utils.checknaninf(ref):
            import pdb
            pdb.set_trace()

        return ref

        


class McClamrochCorkskrew(QuadTrajectory):
    def __init__(self, upper_lim, lower_lim, nominal_dynamics):
        super(McClamrochCorkskrew, self).__init__(upper_lim, lower_lim, nominal_dynamics, nominal_traj_length = int(10//nominal_dynamics.dt))

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

            rx =      np.array([0.4*t, 0.4  * 1        * np.sin(np.pi*t), 0.6  * 1        * np.cos(np.pi*t)])
            rx_dot =  np.array([  0.4, 0.4  * np.pi    * np.cos(np.pi*t), -0.6 * np.pi    * np.sin(np.pi*t)])
            rx_2dot = np.array([  0.0, -0.4 * np.pi**2 * np.sin(np.pi*t), -0.6 * np.pi**2 * np.cos(np.pi*t)])
            rx_3dot = np.array([  0.0, -0.4 * np.pi**3 * np.cos(np.pi*t), 0.6  * np.pi**3 * np.sin(np.pi*t)])
            rx_4dot = np.array([  0.0, 0.4  * np.pi**4 * np.sin(np.pi*t), 0.6  * np.pi**4 * np.cos(np.pi*t)])
            
            rb1 =      np.array([ 1        * np.cos(np.pi*t), 1         * np.sin(np.pi*t), 0])
            rb1_dot =  np.array([-np.pi    * np.sin(np.pi*t), np.pi     * np.cos(np.pi*t), 0])
            rb1_2dot = np.array([-np.pi**2 * np.cos(np.pi*t), -np.pi**2 * np.sin(np.pi*t), 0])

            ref.append(np.concatenate([rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot]))

        if utils.checknaninf(ref):
            import pdb
            pdb.set_trace()

        return ref



class SetPoint(QuadTrajectory):
    def __init__(self, upper_lim, lower_lim, nominal_dynamics):
        super(SetPoint, self).__init__(upper_lim, lower_lim, nominal_dynamics, nominal_traj_length = int(10//nominal_dynamics.dt))

    def __call__(self, x):

        ref = []
        for i in range(self.nominal_traj_length):
            t = i*self._nominal_dynamics.dt

            rx =      np.array([0,0,0])
            rx_dot =  np.array([0,0,0])
            rx_2dot = np.array([0,0,0])
            rx_3dot = np.array([0,0,0])
            rx_4dot = np.array([0,0,0])
            
            rb1 =      np.array([1,0,0])
            rb1_dot =  np.array([0,0,0])
            rb1_2dot = np.array([0,0,0])

            ref.append(np.concatenate([rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot]))

        if utils.checknaninf(ref):
            import pdb
            pdb.set_trace()

        return ref




class RandomMotions(QuadTrajectory):
    def __init__(self, upper_lim, lower_lim, nominal_dynamics, nominal_traj_length=50):
        super(RandomMotions, self).__init__(upper_lim, lower_lim, nominal_dynamics, nominal_traj_length)

    def __call__(self, x):


        p0, R0, v0, W0 = self._nominal_dynamics.split_state(x)
        dx = self._nominal_dynamics(x, np.zeros((4,)))
        dp0, dR0, _, _ = self._nominal_dynamics.split_state(dx)
        b10 = R0[:,1]
        db10 = dR0[:,1]

        dt = self._nominal_dynamics.dt


        ref = []
        rx =      p0
        rx_dot =  v0
        rx_2dot = np.array([0,0,0])
        rx_3dot = np.array([0,0,0])
        rx_4dot = np.array([0,0,0])
        rb1_2dot = np.random.uniform(-1, 1, (3,))
        
        rb1 =      np.reshape(b10, (3,))
        rb1_dot =  np.reshape(db10, (3,))
        rb1_2dot = np.random.uniform(-1, 1, (3,))

        ref.append(np.concatenate([rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot]))

        for i in range(self.nominal_traj_length):

            rx0, rx_dot0, rx_2dot0, rx_3dot0, rx_4dot0, rb10, rb1_dot0, rb1_2dot0 = self.split_ref(ref[-1])

            rx =      rx0 + dt*rx_dot0
            rx_dot =  rx_dot0 + dt*rx_2dot0
            rx_2dot = rx_2dot0 + dt*rx_3dot0
            rx_3dot = rx_3dot0 + dt*rx_4dot0
            rx_4dot = rx_4dot0
            
            rb1 =      rb10 + dt*rb1_dot0
            rb1_dot =  rb1_dot0 + dt*rb1_2dot0
            rb1_2dot = rb1_2dot0

            ref.append(np.concatenate([rx, rx_dot, rx_2dot, rx_3dot, rx_4dot, rb1, rb1_dot, rb1_2dot]))

        if utils.checknaninf(ref):
            import pdb
            pdb.set_trace()

        return ref
        