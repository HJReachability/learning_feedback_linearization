import torch
import numpy as np
from scipy.linalg import block_diag
import math

from dynamics import Dynamics

class Quadrotor12D(Dynamics):
    def __init__(self, mass, Ix, Iy, Iz, time_step=0.05):
        self._mass = mass
        self._Ix = Ix
        self._Iy = Iy
        self._Iz = Iz
        super(Quadrotor12D, self).__init__(12, 11, 3, 3, time_step)

    def __call__(self, x0, u):
        """
        Compute xdot from x, u. Please refer to:
        https://ieeexplore.ieee.org/abstract/document/5164788
        for a full derivation of the dynamics. State is laid out as follows:
               0  1  2  3      4    5     6     7     8     9   10 11
        ` x = [x, y, z, theta, phi, xdot, ydot, zdot, zeta, xi, q, r] `
        ` u = [u1, u2, u3] `
        ` y = [x, y, z] `
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
        dx = x0[5, 0]
        dy = x0[6, 0]
        dz = x0[7, 0]
        zeta = x0[8, 0]
        xi = x0[9, 0]
        q = x0[10, 0]
        r = x0[11, 0]

        # Set psi, p to zero.
        psi = p = 0.0

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
            [g17 * zeta],
            [g18 * zeta],
            [g19 * zeta - g],
            [xi],
            [0.0],
            [0.0],
            [0.0]
        ])

        # Control coefficient matrix. From pp. 34 of the linked document above.
        # NOTE: Ix/y/z might be out of order, but it's consistent with MATLAB so
        # not changing it for now.
        control_coefficient_matrix = np.zeros((self.xdim, self.udim))
        control_coefficient_matrix[9, 0] = 1.0
        control_coefficient_matrix[10, 1] = 1.0 / self._Iy
        control_coefficient_matrix[11, 2] = 1.0 / self._Iz

        xdot = control_coefficient_matrix @ u + drift_term
        return xdot

    def observation(self, x):
        """ Compute y from x. """
        return np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]]])

    def wrap_angles(self, x):
        """ Wrap angles to [-pi, pi]. """
        theta = (x[3, 0] + np.pi) % (2.0 * np.pi) - np.pi
        phi = (x[4, 0] + np.pi) % (2.0 * np.pi) - np.pi

        wrapped_x = x.copy()
        wrapped_x[3, 0] = theta
        wrapped_x[4, 0] = phi
        return wrapped_x

    def preprocess_state(self, x):
        """ Preprocess states for input to learned components. """
        if isinstance(x, torch.Tensor):
            preprocessed_x = torch.zeros(self.preprocessed_xdim)
            cos = torch.cos
            sin = torch.sin
        else:
            preprocessed_x = np.zeros(self.preprocessed_xdim)
            cos = np.cos
            sin = np.sin


        preprocessed_x[0] = cos(x[3])
        preprocessed_x[1] = sin(x[3])
        preprocessed_x[2] = cos(x[4])
        preprocessed_x[3] = sin(x[4])
        preprocessed_x[4] = x[5]
        preprocessed_x[5] = x[6]
        preprocessed_x[6] = x[7]
        preprocessed_x[7] = x[8]
        preprocessed_x[8] = x[9]
        preprocessed_x[9] = x[10]
        preprocessed_x[10] = x[11]

        return preprocessed_x

    def observation_distance(self, y1, y2, norm):
        """ Compute a distance metric on the observation space. """
        if norm == 1:
            dx = abs(y1[0, 0] - y2[0, 0])
            dy = abs(y1[1, 0] - y2[1, 0])
            dz = abs(y1[2, 0] - y2[2, 0])
            return dx + dy + dz
        elif norm == 2:
            dx = abs(y1[0, 0] - y2[0, 0])**2
            dy = abs(y1[1, 0] - y2[1, 0])**2
            dz = abs(y1[2, 0] - y2[2, 0])**2
            return np.sqrt(dx + dy + dz)

        print("You dummy. Bad norm.")
        return np.inf

    def linear_system_state_delta(self, y_ref, y_obs):
        """ Compute a distance metric on the linear system state space. """
        delta = y_obs - y_ref
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
        dx = x0[5, 0]
        dy = x0[6, 0]
        dz = x0[7, 0]
        zeta = x0[8, 0]
        xi = x0[9, 0]
        q = x0[10, 0]
        r = x0[11, 0]

        # Set psi, p to zero.
        psi = p = 0.0

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
            [-(q*zeta*cos(phi)*sin(theta) - xi*cos(phi)*cos(theta) + r*zeta*cos(theta)*sin(phi))/m]
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

        A = block_diag(A1, A1, A1)

        # Construct B.
        B1 = np.zeros((4, 3))
        B1[3, 0] = 1.0

        B2 = np.zeros((4, 3))
        B2[3, 1] = 1.0

        B3 = np.zeros((4, 3))
        B3[3, 2] = 1.0

        B = np.concatenate([B1, B2, B3], axis = 0)

        # Construct C.
        C = np.zeros((3, 12))
        C[0, 0] = 1.0
        C[1, 4] = 1.0
        C[2, 8] = 1.0

        return A, B, C

    def _f_q(self, x0):
        """ This is \alpha(x) on pp. 31. """
        return -np.linalg.inv(self._Delta_q(x0)) @ self._b_q(x0)

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
        dx = x0[5, 0]
        dy = x0[6, 0]
        dz = x0[7, 0]
        zeta = x0[8, 0]
        xi = x0[9, 0]
        q = x0[10, 0]
        r = x0[11, 0]

        # Set psi, p to zero.
        psi = p = 0.0

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        old_delta = np.array([
            [  (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))/m, (zeta*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)))/(Ix*m), (zeta*cos(phi)*cos(psi)*cos(theta))/(Iy*m),  (zeta*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta)))/(Iz*m)],
            [ -(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta))/m, (zeta*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)))/(Ix*m), (zeta*cos(phi)*cos(theta)*sin(psi))/(Iy*m), -(zeta*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)))/(Iz*m)],
            [                               (cos(phi)*cos(theta))/m,                                                                0,         -(zeta*cos(phi)*sin(theta))/(Iy*m),                                -(zeta*cos(theta)*sin(phi))/(Iz*m)],
            [                                                     0,                                                             1/Ix,                                          0,                                                                 0],
        ])

#        return old_delta[[0, 2, 3], :][:, :-1]
        return old_delta[:-1, :][:, [0, 2, 3]]

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
        dx = x0[5, 0]
        dy = x0[6, 0]
        dz = x0[7, 0]
        zeta = x0[8, 0]
        xi = x0[9, 0]
        q = x0[10, 0]
        r = x0[11, 0]

        # Set psi, p to zero.
        psi = p = 0.0

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

        return np.array([f_q1, f_q2, f_q3])
