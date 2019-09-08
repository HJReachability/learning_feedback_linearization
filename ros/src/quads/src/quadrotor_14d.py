#import torch
import numpy as np
from scipy.linalg import block_diag
import math

from dynamics import Dynamics

class Quadrotor14D(Dynamics):
    def __init__(self, mass, Ix, Iy, Iz, time_step=0.05):
        self._mass = mass
        self._Ix = Ix
        self._Iy = Iy
        self._Iz = Iz
        super(Quadrotor14D, self).__init__(14, 14, 4, 4, time_step)

    def __call__(self, x0, u):
        """
        Compute xdot from x, u. Please refer to:
        https://ieeexplore.ieee.org/abstract/document/5164788
        for a full derivation of the dynamics. State is laid out as follows:
        ` x = [x, y, z, psi, theta, phi, xdot, ydot, zdot, zeta, xi, p, q, r] `
        ` u = [u1, u2, u3, u4] `
        ` y = [x, y, z, psi] `
        """
        m = self._mass
        Ix = self._Ix
        Iy = self._Iy
        Iz = self._Iz

        # Unpack x.
        x = x0[0, 0]
        y = x0[1, 0]
        z = x0[2, 0]
        theta = x0[3, 0]
        phi = x0[4, 0]
        psi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        q = x0[11, 0]
        r = x0[12, 0]
        p = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        # Gravity.
        g = 9.81

        # Drift term.
        g17 = (1.0 / m) * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta))
        g18 = (1.0 / m) * (-cos(psi) * sin(phi) + cos(phi) * sin(psi) * sin(theta))
        g19 = (1.0 / m) * (cos(phi) * cos(theta))

        drift_term = np.array([
            [dx],
            [dy],
            [dz],
            [q],
            [r],
            [p],
            [g17 * zeta],
            [g18 * zeta],
            [g19 * zeta - g],
            [xi],
            [0.0],
            [0.0],
            [0.0],
            [0.0]
        ])

        # Control coefficient matrix. From pp. 34 of the linked document above.
        # NOTE: Ix/y/z might be out of order, but it's consistent with MATLAB so
        # not changing it for now.
        control_coefficient_matrix = np.zeros((self.xdim, self.udim))
        control_coefficient_matrix[10, 0] = 1.0
        control_coefficient_matrix[11, 1] = 1.0 / self._Ix
        control_coefficient_matrix[12, 2] = 1.0 / self._Iy
        control_coefficient_matrix[13, 3] = 1.0 / self._Iz

        #control_coefficient_matrix[11, 2] = 1.0 / self._Iy
        #control_coefficient_matrix[12, 1] = 1.0 / self._Ix
        #control_coefficient_matrix[13, 3] = 1.0 / self._Iz

        xdot = np.dot(control_coefficient_matrix, u) + drift_term
        return xdot

    def observation(self, x):
        """ Compute y from x. """
        return np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]], [x[5, 0]]])

    def wrap_angles(self, x):
        """ Wrap angles to [-pi, pi]. """
        psi = (x[5, 0] + np.pi) % (2.0 * np.pi) - np.pi
        theta = (x[3, 0] + np.pi) % (2.0 * np.pi) - np.pi
        phi = (x[4, 0] + np.pi) % (2.0 * np.pi) - np.pi

        wrapped_x = x.copy()
        wrapped_x[5, 0] = psi
        wrapped_x[3, 0] = theta
        wrapped_x[4, 0] = phi
        return wrapped_x

    def observation_distance(self, y1, y2, norm):
        """ Compute a distance metric on the observation space. """
        if norm == 1:
            # HACK only care about z subsystem.
            return np.abs(self.linear_system_state_delta(y1, y2))[8:12,].sum()
#            return np.abs(self.linear_system_state_delta(y1, y2)).sum()
        elif norm == 2:
            # HACK only care about z subsystem.
            delta = self.linear_system_state_delta(y1, y2)
            return np.sqrt(np.multiply(delta, delta)[8:12,].sum())
#            return np.sqrt(np.multiply(delta, delta).sum())

        print("You dummy. Bad norm.")
        return np.inf

    def linear_system_state_delta(self, y_ref, y_obs):
        """ Compute a distance metric on the linear system state space. """
        delta = y_obs - y_ref
        delta[12, 0] = (delta[12, 0] + np.pi) % (2.0 * np.pi) - np.pi
        return delta

    def observation_delta(self, y_ref, y_obs):
        """ Compute a distance metric on the observation space. """

        delta=np.zeros(y_ref.shape)
        delta[0,:]=y_obs[0,:]-y_ref[0,:]
        delta[1,:]=y_obs[1,:]-y_ref[1,:]
        delta[2,:]=y_obs[2,:]-y_ref[2,:]

        des1=y_obs[3,:]
        ref1=y_ref[3,:]

        term1=(des1 - ref1 + np.pi) % (2.0 * np.pi) - np.pi
        term2=(ref1- des1 + np.pi) % (2.0 * np.pi) - np.pi

        delta[3,:]=np.multiply(np.abs(term1)<np.abs(term2),term1)-np.multiply((np.abs(term1)>=np.abs(term2)),term2)

        return delta


    def feedback_linearize(self):
        """
        Computes feedback linearization of this system.
        Returns M matrix and w vector (both functions of state):
                       ``` u = M(x) v + f(x) ```

        :return: M(x) and f(x) as functions
        :rtype: np.array(np.array), np.array(np.array)
        """
        return self._M_q, self._f_q

    def linearized_system_state(self, x0):
        """
        Computes linearized system state `z` from `x` (pp. 35). Please refer to
        the file `quad_sym.m` in which we use MATLAB's symbolic toolkit to
        derive this ungodly mess.
        """
        m = self._mass
        Ix = self._Ix
        Iy = self._Iy
        Iz = self._Iz

        # Unpack x.
        x = x0[0, 0]
        y = x0[1, 0]
        z = x0[2, 0]
        theta = x0[3, 0]
        phi = x0[4, 0]
        psi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        q = x0[11, 0]
        r = x0[12, 0]
        p = x0[13, 0]

        # Gravity.
        g = 9.81

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        return np.array([
            [x],
            [dx],
            [(zeta*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)))/m],
            [(xi*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)))/m + (p*zeta*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)))/m + (r*zeta*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta)))/m + (q*zeta*cos(phi)*cos(psi)*cos(theta))/m],
            [y],
            [dy],
            [-(zeta*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)))/m],
            [(p*zeta*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)))/m - (xi*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)))/m - (r*zeta*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)))/m + (q*zeta*cos(phi)*cos(theta)*sin(psi))/m],
            [z],
            [dz],
            [(zeta*cos(phi)*cos(theta))/m - g],
            [-(q*zeta*cos(phi)*sin(theta) - xi*cos(phi)*cos(theta) + r*zeta*cos(theta)*sin(phi))/m],
            [psi],
            [p]
        ])

    def linearized_system(self):
        """
        Return A, B, C matrices of linearized system from pp. 36, i.e.
             ```
             \dot z = A z + B v
             y = C z
             ```
        """
        # Construct A.
        A1 = np.zeros((4, 4))
        A1[0, 1] = 1.0
        A1[1, 2] = 1.0
        A1[2, 3] = 1.0

        A2 = np.zeros((2, 2))
        A2[0, 1] = 1.0

        A = block_diag(A1, A1, A1, A2)

        # Construct B.
        B1 = np.zeros((4, 4))
        B1[3, 0] = 1.0

        B2 = np.zeros((4, 4))
        B2[3, 1] = 1.0

        B3 = np.zeros((4, 4))
        B3[3, 2] = 1.0

        B4 = np.zeros((2, 4))
        B4[1, 3] = 1.0

        B = np.concatenate([B1, B2, B3, B4], axis = 0)

        # Construct C.
        C = np.zeros((4, 14))
        C[0, 0] = 1.0
        C[1, 4] = 1.0
        C[2, 8] = 1.0
        C[3, 12] = 1.0

        return A, B, C

    def _f_q(self, x0):
        """ This is \alpha(x) on pp. 31. """
        return -np.dot(np.linalg.inv(self._Delta_q(x0)), self._b_q(x0))

    def _M_q(self, x0):
        """ This is \beta(x) on pp. 31. """
        return np.linalg.inv(self._Delta_q(x0))

    def _Delta_q(self, x0):
        """
        v-coefficient matrix in feedback linearization controller. Please refer
        to the file `quad_sym.m` in which we use MATLAB's symbolic toolkit to
        derive this ungodly mess.
        """
        m = self._mass
        Ix = self._Ix
        Iy = self._Iy
        Iz = self._Iz

        # Unpack x.
        x = x0[0, 0]
        y = x0[1, 0]
        z = x0[2, 0]
        theta = x0[3, 0]
        phi = x0[4, 0]
        psi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        q = x0[11, 0]
        r = x0[12, 0]
        p = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        return np.array([[  (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))/m, (zeta*cos(phi)*cos(psi)*cos(theta))/(Ix*m),  (zeta*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta)))/(Iy*m), (zeta*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)))/(Iz*m)],
                         [ -(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta))/m, (zeta*cos(phi)*cos(theta)*sin(psi))/(Ix*m), -(zeta*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)))/(Iy*m), (zeta*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)))/(Iz*m)],
                         [                               (cos(phi)*cos(theta))/m,         -(zeta*cos(phi)*sin(theta))/(Ix*m),                                -(zeta*cos(theta)*sin(phi))/(Iy*m),                                                                0],
                         [                                                     0,                                          0,                                                                 0,                                                             1/Iz]
        ])

    def _b_q(self, x0):
        """
        Drift term in feedback linearization controller. Please refer to
        the file `quad_sym.m` in which we use MATLAB's symbolic toolkit to
        derive this ungodly mess.
        """
        m = self._mass
        Ix = self._Ix
        Iy = self._Iy
        Iz = self._Iz

        # Unpack x.
        x = x0[0, 0]
        y = x0[1, 0]
        z = x0[2, 0]
        theta = x0[3, 0]
        phi = x0[4, 0]
        psi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        q = x0[11, 0]
        r = x0[12, 0]
        p = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        # First term.
        f_q1 = [
            -(p**2*zeta*sin(phi)*sin(psi) - 2*r*xi*cos(phi)*sin(psi) + r**2*zeta*sin(phi)*sin(psi) - 2*p*xi*cos(psi)*sin(phi) + p**2*zeta*cos(phi)*cos(psi)*sin(theta) + q**2*zeta*cos(phi)*cos(psi)*sin(theta) + r**2*zeta*cos(phi)*cos(psi)*sin(theta) - 2*p*r*zeta*cos(phi)*cos(psi) - 2*q*xi*cos(phi)*cos(psi)*cos(theta) + 2*p*xi*cos(phi)*sin(psi)*sin(theta) + 2*r*xi*cos(psi)*sin(phi)*sin(theta) + 2*p*q*zeta*cos(phi)*cos(theta)*sin(psi) + 2*q*r*zeta*cos(psi)*cos(theta)*sin(phi) - 2*p*r*zeta*sin(phi)*sin(psi)*sin(theta))/m
        ]

        # Second term.
        f_q2 = [
            (2*p*xi*sin(phi)*sin(psi) + p**2*zeta*cos(psi)*sin(phi) + r**2*zeta*cos(psi)*sin(phi) - 2*r*xi*cos(phi)*cos(psi) - p**2*zeta*cos(phi)*sin(psi)*sin(theta) - q**2*zeta*cos(phi)*sin(psi)*sin(theta) - r**2*zeta*cos(phi)*sin(psi)*sin(theta) + 2*p*r*zeta*cos(phi)*sin(psi) + 2*p*xi*cos(phi)*cos(psi)*sin(theta) + 2*q*xi*cos(phi)*cos(theta)*sin(psi) - 2*r*xi*sin(phi)*sin(psi)*sin(theta) + 2*p*q*zeta*cos(phi)*cos(psi)*cos(theta) - 2*p*r*zeta*cos(psi)*sin(phi)*sin(theta) - 2*q*r*zeta*cos(theta)*sin(phi)*sin(psi))/m
        ]

        # Third term.
        f_q3 = [
            -(zeta*cos(phi)*cos(theta)*q**2 - 2*zeta*sin(phi)*sin(theta)*q*r + 2*xi*cos(phi)*sin(theta)*q + zeta*cos(phi)*cos(theta)*r**2 + 2*xi*cos(theta)*sin(phi)*r)/m
        ]

        # Fourth term.
        f_q4 = [
            0.0
        ]

        return np.array([f_q1, f_q2, f_q3, f_q4])
