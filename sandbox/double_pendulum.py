import torch
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patches import ConnectionPatch
from matplotlib import cm
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
        self.A,self.B=self.linearized_system()
        

        super(DoublePendulum, self).__init__(4, 6, 2, 2, time_step)

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

    def linearized_system(self):
    
        A=np.zeros((4,4))
        A[0,1]=1
        A[2,3]=1
        
        B=np.zeros((4,2))
        B[1,0]=1
        B[3,1]=1

        return A,B

    def linearized_system_state(self,x):

         return x

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

    def render(self,x,speed=0.01,path=0,t=1):

        blues=cm.get_cmap(name='Blues', lut=None)
        reds=cm.get_cmap(name='Reds', lut=None)
        greys=cm.get_cmap(name='Greys', lut=None)

        if self.fig is None:
            self.fig=plt.figure()
            self.gca=self.fig.gca()
        
        if path:
            self.gca.cla()

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim([-1.1*self._length1-1.1*self._length2,1.1*self._length1+1.1*self._length2])
        plt.ylim([-1.1*self._length1-1.1*self._length2,1.1*self._length1+1.1*self._length2])

        position_joint1=np.array([self._length1*np.sin(x[0,0]),-self._length1*np.cos(x[0,0])])
        position_joint2=np.array([self._length2*np.sin(x[2,0]),-self._length2*np.cos(x[2,0])])+position_joint1
        
        if not path:
            plt.plot([0,position_joint1[0]],[0,position_joint1[1]], color=greys(t), lw=2,zorder=1)
            plt.plot([position_joint1[0],position_joint2[0]],[position_joint1[1],position_joint2[1]],color=greys(t),lw=2,zorder=1)


        self.gca.add_patch(Circle(position_joint1, 0.1*self._length1,facecolor=reds(t),ec='w',zorder=2))
        self.gca.add_patch(Circle(position_joint2, 0.1*self._length2,facecolor=blues(t),ec='w',zorder=2))

        
        plt.pause(speed)

    def wrap_angles(self, x):
        """ Wrap angles to [-pi, pi]. """
        theta1 = (x[0, 0] + np.pi) % (2.0 * np.pi) - np.pi
        theta2 = (x[2, 0] + np.pi) % (2.0 * np.pi) - np.pi
        return np.array([[theta1], [x[1, 0]], [theta2], [x[3, 0]]])

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

        preprocessed_x[0] = cos(x[0])
        preprocessed_x[1] = sin(x[0])
        preprocessed_x[2] = x[1]
        preprocessed_x[3] = cos(x[2])
        preprocessed_x[4] = sin(x[2])
        preprocessed_x[5] = x[3]
        return preprocessed_x

    def observation_distance(self, y1, y2,norm):
        """ Compute a distance metric on the observation space. """
        if norm==1:
            dtheta1 = min(
                abs((y1[0, 0] - y2[0, 0] + np.pi) % (2.0 * np.pi) - np.pi),
                abs((y2[0, 0] - y1[0, 0] + np.pi) % (2.0 * np.pi) - np.pi))
            dtheta2 = min(
                abs((y1[1, 0] - y2[1, 0] + np.pi) % (2.0 * np.pi) - np.pi),
                abs((y2[1, 0] - y1[1, 0] + np.pi) % (2.0 * np.pi) - np.pi))
        elif norm==2:
            dtheta1 = min(
                abs((y1[0, 0] - y2[0, 0] + np.pi) % (2.0 * np.pi) - np.pi)**2,
                abs((y2[0, 0] - y1[0, 0] + np.pi) % (2.0 * np.pi) - np.pi)**2)
            dtheta2 = min(
                abs((y1[1, 0] - y2[1, 0] + np.pi) % (2.0 * np.pi) - np.pi)**2,
                abs((y2[1, 0] - y1[1, 0] + np.pi) % (2.0 * np.pi) - np.pi)**2)
        return dtheta1 + dtheta2

    def observation_delta(self, y_ref, y_obs):
        """ Compute a distance metric on the observation space. """

        delta=np.zeros(y_ref.shape)
        delta[1,:]=y_obs[1,:]-y_ref[1,:]
        delta[3,:]=y_obs[3,:]-y_ref[3,:]

        des1=y_obs[0,:]
        des2=y_obs[2,:]

        ref1=y_ref[0,:]
        ref2=y_ref[2,:]

        term1=(des1 - ref1 + np.pi) % (2.0 * np.pi) - np.pi
        term2=(ref1- des1 + np.pi) % (2.0 * np.pi) - np.pi
        term3=(des2 - ref2 + np.pi) % (2.0 * np.pi) - np.pi
        term4=(ref2 - des2 + np.pi) % (2.0 * np.pi) - np.pi

        delta[0,:]=np.multiply(np.abs(term1)<np.abs(term2),term1)-np.multiply((np.abs(term1)>=np.abs(term2)),term2)
        delta[2,:]=np.multiply(np.abs(term3)<np.abs(term4),term3)-np.multiply(np.abs(term3)>=np.abs(term4),term4)

        return delta
