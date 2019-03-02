import torch
import numpy as np
from scipy.linalg import solve_continuous_are
#import control


from double_pendulum import DoublePendulum
from reinforce import Reinforce
from feedback_linearization import FeedbackLinearization
from logger import *
import matplotlib.pyplot as plt
from plotter import Plotter


filename = "./logs/double_pendulum_3_10_0.100000_0.001000_50_20_dyn_1.050000_0.950000_1.000000_1.000000.pkl"

# Plot everything.
# plotter = Plotter(filename)
# plotter.plot_scalar_fields(["mean_return"])
# plt.pause(0.1)

fp =  fp = open(filename, "rb")
log = dill.load(fp)

fb_law=log['feedback_linearization'][0]


def get_Linear_System():

    A=np.zeros((4,4))
    A[0,1]=1
    A[2,3]=1

    B=np.zeros((4,2))
    B[1,0]=1
    B[3,1]=1

    return A,B



def solve_lqr(A,B,Q,R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    #   K,S,E=control.lqr(A,B,Q,R)
    return K



linear_fb=1
nominal=0
T=1000
to_render=1
check_energy=0
speed=0.001

# Create a double pendulum. And a bad one
mass1 = 1.0
mass2 = 1.0
length1 = 1.0
length2 = 1.0
time_step = 0.02
friction_coeff=0.5
dyn = DoublePendulum(mass1, mass2, length1, length2, time_step,friction_coeff)
bad_dyn = DoublePendulum(.9* mass1, 1.5 * mass2, length1, length2, time_step,friction_coeff)

# LQR Parameters and dynamics
q=10.0
r=1.0
A,B=get_Linear_System()
Q=q*np.diag([1.0,0,1.0,0])
R=r*np.eye(2)

#Get Linear Feedback policies
K=solve_lqr(A,B,Q,R)

#Get random initial state



reference=np.zeros((4,T))
reference[0,:]= np.pi #2*np.pi*np.sin(np.linspace(0,T*time_step,T))
reference[2,:]= np.pi #2*np.pi*np.cos(np.linspace(0,T*time_step,T))

learned_path=np.zeros((4,T+1))
learned_controls_path=np.zeros((2,T))

nominal_path=np.zeros((4,T+1))
nominal_controls_path=np.zeros((2,T))
x0=0.5*np.ones((4,1))

if linear_fb:
    x=x0.copy()
    for t in range(T):
        y=dyn.observation(x)
        ydot=dyn.observation_dot(x)

        desired_linear_system_state=np.zeros((4,1))
        desired_linear_system_state[0,0]=y[0,0]
        desired_linear_system_state[1,0]=ydot[0,0]
        desired_linear_system_state[2,0]=y[1,0]
        desired_linear_system_state[3,0]=ydot[1,0]
        ref=np.reshape(reference[:,t],(4,1))
        #np.concatenate([y,ydot],axis=0)

        v=-1*K @ (desired_linear_system_state-ref)
#        control= np.zeros((2,1))
#        control = fb_law.feedback(x,v).detach().numpy()
        control = bad_dyn._M_q(x) @ v + bad_dyn._f_q(x)

        learned_controls_path[:,t]=control[:,0] #.detach().numpy()
        x=dyn.integrate(x,control) #.detach().numpy())
        learned_path[:,t+1]=(desired_linear_system_state)[:,0]
        if check_energy:
            print(dyn.total_energy(x))
        if to_render:
            dyn.render(x,speed)

if nominal:

    x=x0.copy()
    for t in range(T):
        y=dyn.observation(x)
        ydot=dyn.observation_dot(x)

        desired_linear_system_state=np.zeros((4,1))
        desired_linear_system_state[0,0]=y[0,0]
        desired_linear_system_state[1,0]=ydot[0,0]
        desired_linear_system_state[2,0]=y[1,0]
        desired_linear_system_state[3,0]=ydot[1,0]
        ref=np.reshape(reference[:,t],(4,1))

        v=-1*K @ (desired_linear_system_state-ref)
        control=bad_dyn._M_q(x) @v + bad_dyn._f_q(x)
        nominal_controls_path[:,t]=control[:,0]
        x=dyn.integrate(x,control)
        nominal_path[:,t+1]=(desired_linear_system_state)[:,0]

    plt.plot(nominal_path[0,:],'r')
    plt.plot(nominal_path[2,:],'r')
    plt.plot(reference[0,:],'b')
    plt.plot(reference[2,:],'b')
    plt.show()
