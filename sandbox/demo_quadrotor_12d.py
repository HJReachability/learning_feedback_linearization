import torch
import numpy as np
from scipy.linalg import solve_continuous_are

from quadrotor_12d import Quadrotor12D
from reinforce import Reinforce
from feedback_linearization import FeedbackLinearization
from logger import *
import matplotlib.pyplot as plt
from plotter import Plotter


#filename="./logs/quadrotor_14d_Reinforce_2x32_std1.000000_lr0.001000_kl-1.000000_50_100_fromzero_False_dyn_1.100000_0.900000_0.900000_0.900000_seed_941_norm_2_smallweights_relu.pkl_3"
#filename = "./logs/quadrotor_12d_Reinforce_2x32_std0.250000_lr0.000100_kl-1.000000_25_100_fromzero_False_dyn_0.750000_0.500000_0.500000_0.500000_seed_16_norm_2_smallweights_tanh.pkl_6"
#filename = "./logs/quadrotor_12d_Reinforce_2x32_std0.100000_lr0.001000_kl-1.000000_25_100_fromzero_False_dyn_0.750000_0.500000_0.500000_0.500000_seed_637_norm_2_smallweights_tanh.pkl_6"
#filename = "./logs/quadrotor_12d_Reinforce_2x32_std0.100000_lr0.001000_kl-1.000000_5_500_fromzero_False_dyn_0.750000_0.500000_0.500000_0.500000_seed_375_norm_2_smallweights_tanh.pkl_0"
#filename = "./logs/quadrotor_12d_Reinforce_2x32_std0.250000_lr0.000100_kl-1.000000_25_100_fromzero_False_dyn_0.750000_0.500000_0.500000_0.500000_seed_16_norm_2_smallweights_tanh.pkl_6"
filename = "./logs/quadrotor_12d_Reinforce_2x32_std0.150000_lr0.000100_kl-1.000000_25_100_fromzero_False_dyn_0.750000_0.500000_0.500000_0.500000_seed_441_norm_2_smallweights_tanh.pkl_6"


# Plot everything.
#plotter = Plotter(filename)
#plotter.plot_scalar_fields(["mean_return"])
#plt.pause(0.1)

fp = open(filename, "rb")
log = dill.load(fp)

fb_law=log['feedback_linearization'][0]


def solve_lqr(A,B,Q,R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    #   K,S,E=control.lqr(A,B,Q,R)
    return K

linear_fb=1
nominal=1
ground_truth=1
T=2500
to_render=0
check_energy=0
speed=0.001

# Create a quadrotor.
mass = 1.0
Ix = 1.0
Iy = 1.0
Iz = 1.0
time_step = 0.01
dyn = Quadrotor12D(mass, Ix, Iy, Iz, time_step)

mass_scaling = 0.75
Ix_scaling = 0.5
Iy_scaling = 0.5
Iz_scaling = 0.5
bad_dyn = Quadrotor12D(
    mass_scaling * mass, Ix_scaling * Ix,
    Iy_scaling * Iy, Iz_scaling * Iz, time_step)

#fb_law._M1 = bad_dyn._M_q
#fb_law._f1 = bad_dyn._f_q

# LQR Parameters and dynamics
q=10.0
r=1.0
A, B, C = dyn.linearized_system()
Q=q*np.diag([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
R=r*np.eye(3)

#Get Linear Feedback policies
K=solve_lqr(A,B,Q,R)

#Get random initial state



reference=0.0*np.ones((12,T))
#reference[12, :] = 3.1
#reference[0,:]=0.1*np.linspace(0,T*time_step,T)
#reference[1,:]=0.1*time_step
#reference[4,:]=0.1*np.linspace(0,T*time_step,T)
#reference[5,:]=0.1*time_step
#reference[8,:]=0.1*np.linspace(0,T*time_step,T)
#reference[9,:]=0.1*time_step
#reference[12,:]=0.1*np.linspace(0,T*time_step,T)
#reference[13,:]=0.1*time_step

freq=1.0
reference[0,:]=np.pi*np.sin(freq * np.linspace(0,T*time_step,T))
reference[4,:]=0.5 * np.pi*np.cos(freq * np.linspace(0,T*time_step,T))
reference[8,:]=np.pi*np.cos(freq * np.linspace(0,T*time_step,T))
#reference[3,:]=0.1*np.pi*np.linspace(0, T*time_step, T)

learned_path=np.zeros((12,T+1))
learned_states=np.zeros((12,T+1))
learned_err=np.zeros((12,T+1))
learned_controls_path=np.zeros((3,T))

nominal_path=np.zeros((12,T+1))
nominal_states=np.zeros((12,T+1))
nominal_err=np.zeros((12,T+1))
nominal_controls_path=np.zeros((3,T))

ground_truth_path=np.zeros((12,T+1))
ground_truth_states=np.zeros((12,T+1))
ground_truth_err=np.zeros((12,T+1))
ground_truth_controls_path=np.zeros((3,T))

x0=0.0*np.ones((12,1))
x0[0, 0] = 0.0
x0[1, 0] = 0.5 * np.pi
x0[2, 0] = np.pi
x0[8, 0] = 9.81


if linear_fb:
    x=x0.copy()
    for t in range(T):
        desired_linear_system_state=dyn.linearized_system_state(x)

        ref=np.reshape(reference[:,t],(12,1))

        diff = dyn.linear_system_state_delta(ref, desired_linear_system_state)
        v=-1*K @ diff

        control= fb_law.feedback(x,v)
        learned_controls_path[:,t]=control[:,0].detach().numpy()
        x=dyn.integrate(x,control.detach().numpy())
        learned_err[:,t+1]=(diff)[:,0]
        learned_path[:,t+1]=(desired_linear_system_state)[:,0]
        learned_states[:,t+1]=x[:, 0]

    plt.figure()
    plt.plot(np.linspace(0, T*time_step, T+1),
             learned_path[0,:], '.-r', label="x")
    plt.plot(np.linspace(0, T*time_step, T+1),
             learned_path[4,:], '.-g', label="y")
    plt.plot(np.linspace(0, T*time_step, T+1),
             learned_path[8,:], '.-b', label="z")
    plt.plot(np.linspace(0, T*time_step, T+1),
             learned_states[9,:], '.-y', label="zeta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             learned_states[4,:], '.-m', label="theta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             learned_states[5,:], '.-c', label="phi")
    plt.title("learned")
    plt.legend()

    plt.figure()
    plt.plot(np.linspace(0, T*time_step, T),
             learned_controls_path[0,:], '.-r', label="u1")
    plt.plot(np.linspace(0, T*time_step, T),
             learned_controls_path[1,:], '.-g', label="u2")
    plt.plot(np.linspace(0, T*time_step, T),
             learned_controls_path[2,:], '.-b', label="u3")
    plt.title("learned controls")
    plt.legend()

#        if check_energy:
#            print(dyn.total_energy(x))
#        if to_render:
#            dyn.render(x,speed)

if nominal:

    x=x0.copy()
    dyn.fig=None

    for t in range(T):
        desired_linear_system_state=dyn.linearized_system_state(x)

        ref=np.reshape(reference[:,t],(12,1))

        diff = dyn.linear_system_state_delta(ref, desired_linear_system_state)
        v=-1*K @ diff

        control= bad_dyn._M_q(x) @ v + bad_dyn._f_q(x)
        nominal_controls_path[:,t]=control[:,0]
        x=dyn.integrate(x,control)
        nominal_err[:,t+1]=(diff)[:,0]
        nominal_path[:,t+1]=(desired_linear_system_state)[:,0]
        nominal_states[:,t+1]=x[:, 0]

    plt.figure()
    plt.plot(np.linspace(0, T*time_step, T+1),
             nominal_path[0,:], '.-r', label="x")
    plt.plot(np.linspace(0, T*time_step, T+1),
             nominal_path[4,:], '.-g', label="y")
    plt.plot(np.linspace(0, T*time_step, T+1),
             nominal_path[8,:], '.-b', label="z")
    plt.plot(np.linspace(0, T*time_step, T+1),
             nominal_states[9,:], '.-y', label="zeta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             nominal_states[4,:], '.-m', label="theta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             nominal_states[5,:], '.-c', label="phi")
    plt.title("nominal")
    plt.legend()

#        if to_render:
#            dyn.render(x,speed)

if ground_truth:

    x=x0.copy()
    dyn.fig=None

    for t in range(T):
        desired_linear_system_state=dyn.linearized_system_state(x)

        ref=np.reshape(reference[:,t],(12,1))

        diff = dyn.linear_system_state_delta(ref, desired_linear_system_state)
        v=-1*K @ diff

        control= dyn._M_q(x) @ v + dyn._f_q(x)
#        control = np.zeros((4, 1))
#        control[0, 0] = -0.1 * (diff[8, 0])
        ground_truth_controls_path[:,t]=control[:,0]
        x=dyn.integrate(x,control)
        ground_truth_err[:,t+1]=(diff)[:,0]
        ground_truth_path[:,t+1]=(desired_linear_system_state)[:,0]
        ground_truth_states[:,t+1]=x[:, 0]


    plt.figure()
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_path[0,:], '.-r', label="x")
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_path[4,:], '.-g', label="y")
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_path[8,:], '.-b', label="z")
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_states[9,:], '.-y', label="zeta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_states[4,:], '.-m', label="theta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_states[5,:], '.-c', label="phi")
    plt.legend()
    plt.title("ground truth")

    plt.figure()
    plt.plot(np.linspace(0, T*time_step, T),
             ground_truth_controls_path[0,:], '.-r', label="u1")
    plt.plot(np.linspace(0, T*time_step, T),
             ground_truth_controls_path[1,:], '.-g', label="u2")
    plt.plot(np.linspace(0, T*time_step, T),
             ground_truth_controls_path[2,:], '.-b', label="u3")
    plt.title("ground truth controls")
    plt.legend()


#        if to_render:
#            dyn.render(x,speed)

#plt.plot(np.linalg.norm(nominal_path[:5,:], axis=0),'r')


#plt.plot(np.linspace(0, T*time_step, T+1),
#         np.linalg.norm(learned_path[[0, 4, 8, 12],:], axis=0), 'b')
#plt.plot(reference[2,:],'b')
plt.legend()
plt.show()
