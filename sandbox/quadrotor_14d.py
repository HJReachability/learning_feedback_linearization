import torch
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
        super(Quadrotor14D, self).__init__(14, 17, 4, 4, time_step)

    def __call__(self, x0, u):
        """
        Compute xdot from x, u. Please refer to:
        https://www.kth.se/polopoly_fs/1.588039.1550155544!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf
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
        psi = x0[3, 0]
        theta = x0[4, 0]
        phi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        p = x0[11, 0]
        q = x0[12, 0]
        r = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        # Drift term. From pp. 33 of linked document above.
        g17 = -(1.0 / m) * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta))
        g18 = -(1.0 / m) * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta))
        g19 = -(1.0 / m) * (cos(phi) * cos(theta))

        drift_term = np.array([
            [dx],
            [dy],
            [dz],
            [q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)],
            [q * cos(phi) - r * sin(phi)],
            [p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)],
            [g17 * zeta],
            [g18 * zeta],
            [g19 * zeta],
            [xi],
            [0.0],
            [(Iy - Iz) / Ix * q * r],
            [(Iz - Ix) / Iy * p * r],
            [(Ix - Iy) / Iz * p * q]
        ])

        # Control coefficient matrix. From pp. 34 of the linked document above.
        control_coefficient_matrix = np.zeros((self.xdim, self.udim))
        control_coefficient_matrix[10, 0] = 1.0
        control_coefficient_matrix[11, 1] = 1.0 / self._Ix
        control_coefficient_matrix[12, 2] = 1.0 / self._Iy
        control_coefficient_matrix[13, 3] = 1.0 / self._Iz

        xdot = control_coefficient_matrix @ u + drift_term
        return xdot

    def observation(self, x):
        """ Compute y from x. """
        return np.array([[x[0, 0]], [x[1, 0]], [x[2, 0]], [x[3, 0]]])

    def wrap_angles(self, x):
        """ Wrap angles to [-pi, pi]. """
        psi = (x[3, 0] + np.pi) % (2.0 * np.pi) - np.pi
        theta = (x[4, 0] + np.pi) % (2.0 * np.pi) - np.pi
        phi = (x[5, 0] + np.pi) % (2.0 * np.pi) - np.pi

        wrapped_x = x.copy()
        wrapped_x[3, 0] = psi
        wrapped_x[4, 0] = theta
        wrapped_x[5, 0] = phi
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

        preprocessed_x[0] = x[0]
        preprocessed_x[1] = x[1]
        preprocessed_x[2] = x[2]
        preprocessed_x[3] = cos(x[3])
        preprocessed_x[4] = sin(x[3])
        preprocessed_x[5] = cos(x[4])
        preprocessed_x[6] = sin(x[4])
        preprocessed_x[7] = cos(x[5])
        preprocessed_x[8] = sin(x[5])
        preprocessed_x[9] = x[6]
        preprocessed_x[10] = x[7]
        preprocessed_x[11] = x[8]
        preprocessed_x[12] = x[9]
        preprocessed_x[13] = x[10]
        preprocessed_x[14] = x[11]
        preprocessed_x[15] = x[12]
        preprocessed_x[16] = x[13]
        return preprocessed_x

    def observation_distance(self, y1, y2, norm):
        """ Compute a distance metric on the observation space. """
        if norm == 1:
            dx = abs(y1[0, 0] - y2[0, 0])
            dy = abs(y1[1, 0] - y2[1, 0])
            dz = abs(y1[2, 0] - y2[2, 0])
            dpsi = min(
                abs((y1[3, 0] - y2[3, 0] + np.pi) % (2.0 * np.pi) - np.pi),
                abs((y2[3, 0] - y1[3, 0] + np.pi) % (2.0 * np.pi) - np.pi))
        elif norm == 2:
            dx = abs(y1[0, 0] - y2[0, 0])**2
            dy = abs(y1[1, 0] - y2[1, 0])**2
            dz = abs(y1[2, 0] - y2[2, 0])**2
            dpsi = min(
                abs((y1[3, 0] - y2[3, 0] + np.pi) % (2.0 * np.pi) - np.pi)**2,
                abs((y2[3, 0] - y1[3, 0] + np.pi) % (2.0 * np.pi) - np.pi)**2)
        return dx + dy + dz + dpsi

    def feedback_linearize(self):
        """
        Computes feedback linearization of this system.
        Returns M matrix and w vector (both functions of state):
                       ``` u = M(x) [ f(x) + v ] ```

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
        psi = x0[3, 0]
        theta = x0[4, 0]
        phi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        p = x0[11, 0]
        q = x0[12, 0]
        r = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        return np.array([
            [x],
            [dx],
            [-(zeta*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta)))/m],
            [-(xi*sin(phi)*sin(psi) + p*zeta*cos(phi)*sin(psi) + xi*cos(phi)*cos(psi)*sin(theta) + q*zeta*cos(psi)*cos(theta) - p*zeta*cos(psi)*sin(phi)*sin(theta))/m],
            [y],
            [dy],
            [-(zeta*(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta)))/m],
            [-(xi*cos(psi)*sin(phi) - q*zeta*cos(theta)*sin(psi) - xi*cos(phi)*sin(psi)*sin(theta) + p*zeta*cos(phi)*cos(psi) + p*zeta*sin(phi)*sin(psi)*sin(theta))/m],
            [z],
            [dz],
            [-(zeta*cos(phi)*cos(theta))/m],
            [(q*zeta*sin(theta) - xi*cos(phi)*cos(theta) + p*zeta*cos(theta)*sin(phi))/m],
            [(psi + np.pi) % (2.0 * np.pi) - np.pi],
            [(r*cos(phi) + q*sin(phi))/cos(theta)]
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
        psi = x0[3, 0]
        theta = x0[4, 0]
        phi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        p = x0[11, 0]
        q = x0[12, 0]
        r = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        return np.array([
            [-(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))/m, -(zeta*(cos(phi)*sin(psi) - cos(psi)*sin(phi)*sin(theta)))/(Ix*m), -(zeta*cos(psi)*cos(theta))/(Iy*m), 0],
            [-(cos(psi)*sin(phi) - cos(phi)*sin(psi)*sin(theta))/m, -(zeta*(cos(phi)*cos(psi) + sin(phi)*sin(psi)*sin(theta)))/(Ix*m),  (zeta*cos(theta)*sin(psi))/(Iy*m), 0],
            [-(cos(phi)*cos(theta))/m, (zeta*cos(theta)*sin(phi))/(Ix*m), (zeta*sin(theta))/(Iy*m), 0],
            [0, 0, sin(phi)/(Iy*cos(theta)), cos(phi)/(Iz*cos(theta))]
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
        psi = x0[3, 0]
        theta = x0[4, 0]
        phi = x0[5, 0]
        dx = x0[6, 0]
        dy = x0[7, 0]
        dz = x0[8, 0]
        zeta = x0[9, 0]
        xi = x0[10, 0]
        p = x0[11, 0]
        q = x0[12, 0]
        r = x0[13, 0]

        # Fix sines, cosines, and tangents.
        sin = np.sin
        cos = np.cos
        tan = math.tan

        # First term.
        f_q1 = [(Ix*Iy*p**2*zeta*sin(phi)*sin(psi) + Ix*Iy*q**2*zeta*sin(phi)*sin(psi) + Ix**2*p*r*zeta*cos(psi)*cos(theta) - Iy**2*q*r*zeta*cos(phi)*sin(psi) - 2*Ix*Iy*q*xi*cos(psi)*cos(theta) - 2*Ix*Iy*p*xi*cos(phi)*sin(psi) + Iy**2*q*r*zeta*cos(psi)*sin(phi)*sin(theta) + Ix*Iy*q*r*zeta*cos(phi)*sin(psi) + Iy*Iz*q*r*zeta*cos(phi)*sin(psi) + 2*Ix*Iy*p*xi*cos(psi)*sin(phi)*sin(theta) + Ix*Iy*p**2*zeta*cos(phi)*cos(psi)*sin(theta) + Ix*Iy*q**2*zeta*cos(phi)*cos(psi)*sin(theta) - Ix*Iy*p*r*zeta*cos(psi)*cos(theta) - Ix*Iz*p*r*zeta*cos(psi)*cos(theta) - Ix*Iy*q*r*zeta*cos(psi)*sin(phi)*sin(theta) - Iy*Iz*q*r*zeta*cos(psi)*sin(phi)*sin(theta))/(Ix*Iy*m)
]

        # Second term.
        f_q2 = [(2*Ix*Iy*q*xi*cos(theta)*sin(psi) - Ix**2*p*r*zeta*cos(theta)*sin(psi) - 2*Ix*Iy*p*xi*cos(phi)*cos(psi) - Iy**2*q*r*zeta*cos(phi)*cos(psi) + Ix*Iy*p**2*zeta*cos(psi)*sin(phi) + Ix*Iy*q**2*zeta*cos(psi)*sin(phi) + Ix*Iy*p*r*zeta*cos(theta)*sin(psi) + Ix*Iz*p*r*zeta*cos(theta)*sin(psi) - Iy**2*q*r*zeta*sin(phi)*sin(psi)*sin(theta) - 2*Ix*Iy*p*xi*sin(phi)*sin(psi)*sin(theta) - Ix*Iy*p**2*zeta*cos(phi)*sin(psi)*sin(theta) - Ix*Iy*q**2*zeta*cos(phi)*sin(psi)*sin(theta) + Ix*Iy*q*r*zeta*cos(phi)*cos(psi) + Iy*Iz*q*r*zeta*cos(phi)*cos(psi) + Ix*Iy*q*r*zeta*sin(phi)*sin(psi)*sin(theta) + Iy*Iz*q*r*zeta*sin(phi)*sin(psi)*sin(theta))/(Ix*Iy*m)
]

        # Third term.
        f_q3 = [(2*Ix*Iy*q*xi*sin(theta) - Ix**2*p*r*zeta*sin(theta) + Iy**2*q*r*zeta*cos(theta)*sin(phi) + Ix*Iy*p*r*zeta*sin(theta) + Ix*Iz*p*r*zeta*sin(theta) + 2*Ix*Iy*p*xi*cos(theta)*sin(phi) + Ix*Iy*p**2*zeta*cos(phi)*cos(theta) + Ix*Iy*q**2*zeta*cos(phi)*cos(theta) - Ix*Iy*q*r*zeta*cos(theta)*sin(phi) - Iy*Iz*q*r*zeta*cos(theta)*sin(phi))/(Ix*Iy*m)
]

        # Fourth term.
        f_q4 = [((q*cos(phi) - r*sin(phi))*(p + r*cos(phi)*tan(theta) + q*sin(phi)*tan(theta)))/cos(theta) + (sin(theta)*(r*cos(phi) + q*sin(phi))*(q*cos(phi) - r*sin(phi)))/cos(theta)**2 + (p*q*cos(phi)*(Ix - Iy))/(Iz*cos(theta)) - (p*r*sin(phi)*(Ix - Iz))/(Iy*cos(theta))
]

        return np.array([f_q1, f_q2, f_q3, f_q4])
