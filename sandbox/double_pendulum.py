import torch
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from dynamics import Dynamics

class DoublePendulum(Dynamics):
    def __init__(self, mass1, mass2, length1, length2,
                 time_step=0.05, friction_coeff=0.5):
        self._mass1 = mass1
        self._mass2 = mass2
        self._length1 = length1
        self._length2 = length2
        self.fig = None
        self.friction_coeff = friction_coeff
        self.g = 9.81

        super(DoublePendulum, self).__init__(4, 2, 2, time_step)

    def __call__(self, x, u):
        """
        Compute xdot from x, u.
        State x is ordered as follows: [theta1, theta1 dot, theta2, theta2 dot].
        """
        xdot = np.zeros((self.xdim, 1))
        xdot[0, 0] = x[1, 0]
        xdot[2, 0] = x[3, 0]

        theta_doubledot = self._M_q_inv(x) @ (
            -self._f_q(x) + np.reshape(u, (self.udim, 1)))

        xdot[1, 0] = theta_doubledot[0, 0]
        xdot[3, 0] = theta_doubledot[1, 0]

        return xdot

    def _M_q(self, x):
        """ Mass matrix. """
        term = self._mass2 * self._length2 * self._length1 * np.cos(x[2, 0])

        M = np.zeros((2, 2))
        M[0, 0] = (self._mass1 + self._mass2) * self._length1**2 + \
                  self._mass2 * self._length2**2 + 2.0 * term
        M[0, 1] = self._mass2 * self._length2**2 + term
        M[1, 0] = self._mass2 * self._length2**2 + term
        M[1, 1] = self._mass2 * self._length2**2

        return M

    def _M_q_inv(self, x):
         return np.linalg.inv(self._M_q(x))

    def _f_q(self, x):
        """ Drift term. """
        term = self._mass2 * self._length2 * self._length1
        F = np.zeros((2, 1))
        G = np.zeros((2, 1))

        G[0, 0] = self.g * (
            (self._mass1 + self._mass2) * self._length1 * np.sin(x[0,0]) +
            self._mass2 * self._length2 * np.sin(x[0, 0] + x[2, 0]))
        G[1, 0] = self.g * (
            self._mass2 * self._length2 * np.sin(x[0, 0] + x[2, 0]))

        F[0, 0] = -term * (2.0 * x[1, 0] + x[3, 0]) * np.sin(x[2, 0]) * x[3,0]
        F[1, 0] = term * x[1, 0] * np.sin(x[2, 0]) * x[1, 0]

        friction = np.zeros((2, 1))
        friction[0, 0] = self.friction_coeff * x[1 ,0]
        friction[1, 0] = self.friction_coeff * x[3, 0]

        return F + G + friction

    def energy(self,x):
        G=np.zeros((2,1))

        G[0,0]=self.g*((self._mass1+self._mass2)*self._length1*np.sin(x[0,0])+self._mass2*self._length2*np.sin(x[0,0]+x[2,0]))
        G[1,0]=self.g*(self._mass2*self._length2*np.sin(x[0,0]+x[2,0]))

        M=self._M_q(x)
        qdot=np.zeros((2,1))
        qdot[0,0]=x[1,0]
        qdot[1,0]=x[3,0]
        energy = (0.5* qdot.T @ M @ qdot)
        energy+=-(self._mass1+self._mass2)*self.g*self._length1*np.cos(x[0,0])-self._mass2*self.g*self._length2*np.cos(x[0,0]+x[2,0])

        return energy

    def total_energy(self,x):
        e=self.energy(x)
        e-=self.energy(np.zeros((4,1)))
        return e

    def observation(self, x):
        """ Compute y from x. """
        return np.array([[x[0, 0]], [x[2, 0]]])

    def observation_dot(self, x):
        """ Compute y from x. """
        return np.array([[x[1, 0]], [x[3, 0]]])

    def feedback_linearize(self):
        """
        Computes feedback linearization of this system.
        Returns M matrix and w vector (both functions of state):
                       ``` u = M(x) [ w(x) + v ] ```

        :return: M(x) and w(x) as functions
        :rtype: np.array(np.array), np.array(np.array)
        """
        return self._M_q, self._f_q

    def render(self,x,speed=0.01):
        if self.fig is None:
            self.fig=plt.figure()
            self.gca=self.fig.gca()

        self.gca.cla()

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([-1.1*self._length1-1.1*self._length2,1.1*self._length1+1.1*self._length2])
        plt.ylim([-1.1*self._length1-1.1*self._length2,1.1*self._length1+1.1*self._length2])

        position_joint1=np.array([self._length1*np.sin(x[0,0]),-self._length1*np.cos(x[0,0])])
        position_joint2=np.array([self._length2*np.sin(x[2,0]),-self._length2*np.cos(x[2,0])])+position_joint1

        self.gca.add_patch(Circle(position_joint1, 0.1*self._length1,facecolor='r',ec='k'))
        self.gca.add_patch(Circle(position_joint2, 0.1*self._length2,facecolor='b',ec='k'))
        plt.plot([0,position_joint1[0]],[0,position_joint1[1]],'k')
        plt.plot([position_joint1[0],position_joint2[0]],[position_joint1[1],position_joint2[1]],'k')
        plt.pause(speed)


    def wrap_angles(self, x):
        """ Wrap angles to [-pi, pi]. """
        theta1 = (x[0, 0] + np.pi) % (2.0 * np.pi) - np.pi
        theta2 = (x[2, 0] + np.pi) % (2.0 * np.pi) - np.pi
        return np.array([[theta1], [x[1, 0]], [theta2], [x[3, 0]]])

    def integrate(self, x0, u, dt=None):
        """
        Integrate initial state x0 (applying constant control u)
        over a time interval of self._time_step, using a time discretization
        of dt.
        NOTE: overriding the base class implementation to do angle wrapping.

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
            dt = 0.25 * self._time_step

        t = 0.0
        x = x0.copy()
        while t < self._time_step - 1e-8:
            # Make sure we don't step past T.
            step = min(dt, self._time_step - t)

            # Use Runge-Kutta order 4 integration. For details please refer to
            # https://en.wikipedia.org/wiki/Runge-Kutta_methods.
            k1 = step * self.__call__(x, u)
            k2 = step * self.__call__(x + 0.5 * k1, u)
            k3 = step * self.__call__(x + 0.5 * k2, u)
            k4 = step * self.__call__(x + k3, u)

            x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += step

        return self.wrap_angles(x)
