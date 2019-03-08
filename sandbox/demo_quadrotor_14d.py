import torch
import numpy as np
from scipy.linalg import solve_continuous_are

from quadrotor_14d import Quadrotor14D
from reinforce import Reinforce
from feedback_linearization import FeedbackLinearization
from logger import *
import matplotlib.pyplot as plt
from plotter import Plotter


filename='./logs/quadrotor_14d_Reinforce_3x10_std0.100000_lr0.001000_kl-1.000000_50_25_dyn_0.900000_1.100000_1.100000_1.100000_seed_70.pkl'

# Plot everything.
# plotter = Plotter(filename)
# plotter.plot_scalar_fields(["mean_return"])
# plt.pause(0.1)

fp =  fp = open(filename, "rb")
log = dill.load(fp)

fb_law=log['feedback_linearization'][0]


def solve_lqr(A,B,Q,R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    #   K,S,E=control.lqr(A,B,Q,R)
    return K

linear_fb=1
nominal=1
T=100
to_render=0
check_energy=0
speed=0.001

# Create a quadrotor.
mass = 1.0
Ix = 1.0
Iy = 1.0
Iz = 1.0
time_step = 0.005
dyn = Quadrotor14D(mass, Ix, Iy, Iz, time_step)

mass_scaling = 0.9
Ix_scaling = 1.1
Iy_scaling = 1.1
Iz_scaling = 1.1
bad_dyn = Quadrotor14D(
    mass_scaling * mass, Ix_scaling * Ix,
    Iy_scaling * Iy, Iz_scaling * Iz, time_step)

# LQR Parameters and dynamics
q=100.0
r=1.0
A, B, C = dyn.linearized_system()
Q=q*np.diag([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
R=r*np.eye(4)

#Get Linear Feedback policies
K=solve_lqr(A,B,Q,R)

#Get random initial state



reference=0.0*np.ones((14,T))
reference[0,:]=0.1*np.linspace(0,T*time_step,T)
reference[1,:]=0.1*time_step
reference[4,:]=0.1*np.linspace(0,T*time_step,T)
reference[5,:]=0.1*time_step
reference[8,:]=0.1*np.linspace(0,T*time_step,T)
reference[9,:]=0.1*time_step
reference[12,:]=0.1*np.linspace(0,T*time_step,T)
reference[13,:]=0.1*time_step
#reference[0,:]=np.pi*np.sin(np.linspace(0,T*time_step,T))
#reference[1,:]=np.pi*np.cos(np.linspace(0,T*time_step,T))
#reference[2,:]=np.pi*np.cos(np.linspace(0,T*time_step,T))
#reference[3,:]=0.1*np.pi*np.linspace(0, T*time_step, T)

learned_path=np.zeros((14,T+1))
learned_err=np.zeros((14,T+1))
learned_controls_path=np.zeros((4,T))

nominal_path=np.zeros((14,T+1))
nominal_states=np.zeros((14,T+1))
nominal_err=np.zeros((14,T+1))
nominal_controls_path=np.zeros((4,T))
x0=0.0*np.ones((14,1))
x0[9, 0] = -0.1

if linear_fb:
    x=x0.copy()
    for t in range(T):
        desired_linear_system_state=dyn.linearized_system_state(x)

        ref=np.reshape(reference[:,t],(14,1))

        diff = desired_linear_system_state - ref
        v=-1*K @ diff

        control= fb_law.feedback(x,v)
        learned_controls_path[:,t]=control[:,0].detach().numpy()
        x=dyn.integrate(x,control.detach().numpy())
        learned_err[:,t+1]=(diff)[:,0]
        learned_path[:,t+1]=(desired_linear_system_state)[:,0]

#        if check_energy:
#            print(dyn.total_energy(x))
#        if to_render:
#            dyn.render(x,speed)

if nominal:

    x=x0.copy()
    dyn.fig=None

    for t in range(T):
        desired_linear_system_state=dyn.linearized_system_state(x)

        ref=np.reshape(reference[:,t],(14,1))

        diff = desired_linear_system_state - ref
        v=-1*K @ diff

        control= dyn._M_q(x) @ v + dyn._f_q(x)
        nominal_controls_path[:,t]=control[:,0]
        x=dyn.integrate(x,control)
        nominal_err[:,t+1]=(diff)[:,0]
        nominal_path[:,t+1]=(desired_linear_system_state)[:,0]
        nominal_states[:,t+1]=x[:, 0]
#        if to_render:
#            dyn.render(x,speed)

#plt.plot(np.linalg.norm(nominal_path[:5,:], axis=0),'r')
plt.figure()
plt.plot(np.linspace(0, T*time_step, T+1),
         learned_path[0,:], '.-r', label="x")
plt.plot(np.linspace(0, T*time_step, T+1),
         learned_path[4,:], '.-g', label="y")
plt.plot(np.linspace(0, T*time_step, T+1),
         learned_path[8,:], '.-b', label="z")
plt.plot(np.linspace(0, T*time_step, T+1),
         learned_path[12,:], '.-k', label="psi")
plt.title("learned")
plt.legend()

plt.figure()
plt.plot(np.linspace(0, T*time_step, T+1),
         nominal_path[0,:], '.-r', label="x")
plt.plot(np.linspace(0, T*time_step, T+1),
         nominal_path[4,:], '.-g', label="y")
plt.plot(np.linspace(0, T*time_step, T+1),
         nominal_path[8,:], '.-b', label="z")
plt.plot(np.linspace(0, T*time_step, T+1),
         nominal_path[12,:], '.-k', label="psi")
plt.plot(np.linspace(0, T*time_step, T+1),
         nominal_states[9,:], '.-y', label="zeta")


#plt.plot(np.linspace(0, T*time_step, T+1),
#         np.linalg.norm(learned_path[[0, 4, 8, 12],:], axis=0), 'b')
#plt.plot(reference[2,:],'b')
plt.title("nominal")
plt.legend()
plt.show()