import numpy as np
from scipy.linalg import block_diag
import math

from dynamics import Dynamics

class DiffDrive(Dynamics):
    def __init__(self,time_step=0.01):
        super(DiffDrive, self).__init__(14, 14, 4, 4, time_step)
        self.xdim = 4
        self.udim = 2
        self._time_step = time_step

    def __call__(self, x0, u):
        ''' x: x, y theta, v
            y: x, y, xdot, ydot
            u: a, w

            xdot: vcos(theta), vsin(theta), w, a
            
            Method returns xdot'''
            x = x0.copy()

            xdot = []
            xdot.append(x[3]*np.cos(x[2]))
            xdot.append(x[3]*np.sin(x[2]))
            xdot.append(u[1])
            xdot.append(u[0])
            
            return np.array(xdot)

    def wrap_angles(self, x):
        """ Makes sure theta is between (-pi,pi] """
        x = x0.copy()
        x[2] = x[2]%np.pi
        return x

    def observation_distance(self, y1, y2, norm):
        """ Compute a distance metric on the observation space. """
        if norm == 1:
            return np.abs(self.linear_system_state_delta(y1, y2)).sum()
        elif norm == 2:
            delta = self.linear_system_state_delta(y1, y2)
            return np.sqrt(np.multiply(delta, delta).sum())

        print("You dummy. Bad norm.")
        return np.inf
    def preprocess_state(self, x0):
        x = x0.copy()
        x = list(x)
        x[2] = np.sin(x[2])
        x.append(np.cos(x[2]))
        return np.array(x)


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
        x = x0.copy
        y = []

        theta = x[2]
        velocity = x[3]
        y.append(x[0])
        y.append(x[1])
        y.append(velocity*np.cos(theta))
        y.append(velocity*np.sin(theta))

        return np.array(y)
       

    def linearized_system(self):
        """
        Return A, B, C matrices of linearized system from pp. 36, i.e.
             ```
             \dot z = A z + B v
             y = C z
             ```
        """
        A = np.zeros((4,4))
        A[0,2] = 1
        A[1,3] = 1

        B = np.zeros((4,2))
        B[2,0] = 1
        B[3,1] = 1

        C = np.zeros((2,4))
        C[0,0] = 1
        C[1,1] = 1

        return A,B,C

        

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
        x = x0.copy()
    
        theta = x[2]
        velocity = x[3]

        Q = np.array([[0,0],[0,0],[cos(theta), -velocity*sin(theta)],[sin(theta), velocity*cos(theta)]])
        return Q

    def _b_q(self, x):
        """
        Drift term in feedback linearization controller. Please refer to
        the file `quad_sym.m` in which we use MATLAB's symbolic toolkit to
        derive this ungodly mess.
        """
        x = x0.copy()
        b = np.array([0,0])
        return b
        