import numpy as np
from scipy.linalg import solve_continuous_are
import os.path as osp, time, atexit, os

from quadrotor_14d import Quadrotor14D
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as ani
from matplotlib.gridspec import GridSpec
from plotter import Plotter
import spinup
import spinup.algos.vpg.core as core
import tensorflow as tf
from gym import spaces
from matplotlib import cm

#should be in form ./logs/ and then whatever foldername logger dumped
# filename = 'ppo-v3-yawcontroller'
# filename = 'poly-10-0.33-null-v2-preprocess-largerQ-start25-uscaling0.1-parameternorm-lr5e-5-normscaling0.001'
filename = 'ppo-10-0.33-null-v2-preprocess-largerQ-start25-uscaling0.1'
# filename = 'poly-v3-continuousrollout-originreturn'
# filename = 'poly-v3-continuousrollout-originreturn-dynamicsScaling0.6'
# filename = 'ppo-v3-continuousrollout-originreturn-dynamicsScaling0.6-zeroweights'

filepath="./logs/{}/simple_save".format(filename)

#load tensorflow graph from path
sess = tf.Session()
model = spinup.utils.logx.restore_tf_graph(sess,filepath)
print(model)

#setting sigma variables to zero
sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name("pi/log_std:0"), tf.zeros((20,))))

# Plot everything.
def solve_lqr(A,B,Q,R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    #   K,S,E=control.lqr(A,B,Q,R)
    return K

linear_fb=1
nominal=0
ground_truth=1
T=3000
show = 0
make_plot = 1
make_vid = 0
show_ref = 0
to_render=1
speed=0.01

# Create a quadrotor.
mass = 1.0
Ix = 1.0
Iy = 1.0
Iz = 1.0
time_step = 0.01
dyn = Quadrotor14D(mass, Ix, Iy, Iz, time_step)

bad_scaling = 0.33
bad_dyn = Quadrotor14D(
    bad_scaling * mass, bad_scaling * Ix,
    bad_scaling* Iy, bad_scaling * Iz, time_step)

# LQR Parameters and dynamics
q=10.0
r=1.0
A, B, C = dyn.linearized_system()
Q=q*np.diag([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
R=r*np.eye(4)

#Get Linear Feedback policies
K=solve_lqr(A,B,Q,R)

#Get random initial state
reference=0.0*np.ones((14,T))

# testname = "circle"
# freq= 0.5
# reference[0,:]=np.pi*np.sin(freq * np.linspace(0,T*time_step,T))
# reference[4,:]=0.5 * np.pi*np.cos(freq * np.linspace(0,T*time_step,T))
# reference[8,:]=np.pi*np.cos(freq * np.linspace(0,T*time_step,T))

# testname = "line_w/yaw"
# reference[0,:]=0.1*np.linspace(0,T*time_step,T)
# reference[1,:]=0.1*time_step
# reference[4,:]=0.1*np.linspace(0,T*time_step,T)
# reference[5,:]=0.1*time_step
# reference[8,:]=0.1*np.linspace(0,T*time_step,T)
# reference[9,:]=0.1*time_step
# reference[12,:]=0.1*np.linspace(0,T*time_step,T)
# reference[13,:]=0.1*time_step

testname = "Figure8"
freq=1.0
reference[0,:]=np.pi*np.sin(freq * np.linspace(0,T*time_step,T))
reference[4,:]=np.pi*np.cos(freq /2.0* np.linspace(0,T*time_step,T))
reference[8,:]=5.0-2.0*np.sin(np.linspace(0,T*time_step,T))
reference[12,:]= np.pi/2*np.sin(0.5*np.linspace(0,T*time_step,T))


# testname = "PointyCorkscrew"
# freq=2.0
# reference[0,:]=np.pi*np.sin(freq * np.linspace(0,T*time_step,T))/(freq * np.linspace(0,T*time_step,T)+1)
# reference[4,:]=np.pi*np.cos(freq * np.linspace(0,T*time_step,T))/(freq * np.linspace(0,T*time_step,T)+1)
# reference[8,:]=5.0-5.0*np.linspace(0,T*time_step,T)/(T*time_step)
# reference[12,:]= np.pi*np.sin(0.5*np.linspace(0,T*time_step,T))


learned_path=np.zeros((14,T+1))
learned_states=np.zeros((14,T+1))
learned_err=np.zeros((14,T+1))
learned_controls_path=np.zeros((4,T))

nominal_path=np.zeros((14,T+1))
nominal_states=np.zeros((14,T+1))
nominal_err=np.zeros((14,T+1))
nominal_controls_path=np.zeros((4,T))

ground_truth_path=np.zeros((14,T+1))
ground_truth_states=np.zeros((14,T+1))
ground_truth_err=np.zeros((14,T+1))
ground_truth_controls_path=np.zeros((4,T))

x0=0.0*np.ones((14,1))
x0[0, 0] = reference[0,0]
x0[1, 0] = reference[4,0]
x0[2, 0] = reference[8,0]
x0[9, 0] = 9.81

def preprocess_state(x):
    x[0] = np.sin(x[3])
    x[1] = np.sin(x[4])
    x[2]= np.sin(x[5])
    x[3] = np.cos(x[3])
    x[4] = np.cos(x[4])
    x[5]= np.cos(x[5])
    x.pop(10)
    return x

output_dir = './tests/{}/{}/'.format(filename,testname)
if osp.exists(output_dir):
    print("Warning: Log dir %s already exists! Storing info there anyway."%output_dir)
else:
    os.makedirs(output_dir)

if linear_fb:
    x=x0.copy()
    for t in range(T):
        desired_linear_system_state=dyn.linearized_system_state(x)

        ref=np.reshape(reference[:,t],(14,1))

        diff = desired_linear_system_state - ref
        v=-1*K @ diff
        
        obs = np.array(preprocess_state(x.tolist()))
        # obs = x
    
        u = sess.run(tf.get_default_graph().get_tensor_by_name('pi/add:0'),{tf.get_default_graph().get_tensor_by_name('Placeholder:0'): obs.reshape(1,-1)})
         #output of neural network
        u = 0.1*u[0]
        m2, f2 = np.split(u,[16])

        M = bad_dyn._M_q(x) + np.reshape(m2,(4, 4))

        f = bad_dyn._f_q(x) + np.reshape(f2,(4, 1))

        control = np.matmul(M, v) + f

        #stablizing yaw controller
        # control[3] = v[3]
        
        learned_controls_path[:,t]=control[:,0]
        x=dyn.integrate(x,control)
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
    plt.plot(np.linspace(0, T*time_step, T+1),
             learned_path[12,:], '.-k', label="psi")
    plt.title("learned")
    plt.legend()
    plt.savefig(output_dir + 'learned_state.png')


if nominal:

    x=x0.copy()

    for t in range(T):
        desired_linear_system_state=dyn.linearized_system_state(x)

        ref=np.reshape(reference[:,t],(14,1))

        diff = desired_linear_system_state - ref
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
             nominal_path[12,:], '.-k', label="psi")
    plt.plot(np.linspace(0, T*time_step, T+1),
            nominal_states[9,:], '.-y', label="zeta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             nominal_states[4,:], '.-m', label="theta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             nominal_states[5,:], '.-c', label="phi")
    plt.title("nominal")
    plt.savefig(output_dir + 'nominal_state.png')

if ground_truth:

    x=x0.copy()

    for t in range(T):
        desired_linear_system_state=dyn.linearized_system_state(x)

        ref=np.reshape(reference[:,t],(14,1))

        diff = desired_linear_system_state - ref
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
             ground_truth_path[12,:], '.-k', label="psi")
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_states[9,:], '.-y', label="zeta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_states[4,:], '.-m', label="theta")
    plt.plot(np.linspace(0, T*time_step, T+1),
             ground_truth_states[5,:], '.-c', label="phi")
    plt.legend()
    plt.title("ground truth")
    plt.savefig(output_dir + 'ground_truth_state.png')

    # plt.figure()
    # plt.plot(np.linspace(0, T*time_step, T),
    #          ground_truth_controls_path[0,:], '.-r', label="u1")
    # plt.plot(np.linspace(0, T*time_step, T),
    #          ground_truth_controls_path[1,:], '.-g', label="u2")
    # plt.plot(np.linspace(0, T*time_step, T),
    #          ground_truth_controls_path[2,:], '.-b', label="u3")
    # plt.plot(np.linspace(0, T*time_step, T),
    #          ground_truth_controls_path[3,:], '.-k', label="u4")
    # plt.title("ground truth controls")
    # plt.legend()


    
if show or make_vid:
    fig=plt.figure(figsize=(10,6))
    fig.suptitle(filename + ' ' + testname)
    if make_vid:
        writer = ani.FFMpegWriter(fps=10)
        writer.setup(fig, output_dir + filename + testname + '.mp4', dpi=100)
    
    gs=GridSpec(1,3)
    ax = fig.add_subplot(gs[0,0:2],projection = '3d')
    states = fig.add_subplot(gs[0,2])
    skip=10


    for i in range(len(ground_truth_path[0,:])-1):
        if i%skip==0:
            if i==0:
                last=0
            else:
                last=i-skip

            if nominal:
                ax.plot(nominal_path[0,last:i],nominal_path[4,last:i],nominal_path[8,last:i],'r')
                p1=ax.scatter(nominal_path[0,i],nominal_path[4,i],nominal_path[8,i],c='r', label = 'nominal')
            
            if linear_fb:
                ax.plot(learned_path[0,last:i],learned_path[4,last:i],learned_path[8,last:i],'g')
                p2=ax.scatter(learned_path[0,i],learned_path[4,i],learned_path[8,i],c='g',label='Learned policy')
                states.plot(np.linspace(last*time_step, i*time_step, i-last), learned_states[3,last:i], '.-y', label="learned-theta")
                states.plot(np.linspace(last*time_step, i*time_step, i-last), learned_states[4,last:i], '.-g', label="learned-phi")
                states.plot(np.linspace(last*time_step, i*time_step, i-last), learned_states[5,last:i], '.-b', label="learned-psi")
    
            if ground_truth:
                ax.plot(ground_truth_path[0,last:i],ground_truth_path[4,last:i],ground_truth_path[8,last:i],'b')
                p3=ax.scatter(ground_truth_path[0,i],ground_truth_path[4,i],ground_truth_path[8,i],c='b',label='Ground Truth')
                states.plot(np.linspace(last*time_step, i*time_step, i-last), ground_truth_states[3,last:i], ':y', label="ground-truth-theta")
                states.plot(np.linspace(last*time_step, i*time_step, i-last), ground_truth_states[4,last:i], ':g', label="ground-truth-phi")
                states.plot(np.linspace(last*time_step, i*time_step, i-last), ground_truth_states[5,last:i], ':b', label="ground-truth-psi")

            if show_ref:
                ax.plot(reference[0,last:i],reference[4,last:i],reference[8,last:i],'k')
                p4=ax.scatter(reference[0,i],reference[4,i],reference[8,i],c='k',label='Reference')
        
            # if show:
            #     plt.pause(0.01)

            if make_vid:
                writer.grab_frame()
            ax.legend(loc=0)
            plt.tight_layout(pad = 3, h_pad = 3, w_pad = 3)
            if(i == 0):
                states.legend(loc=0)
            if i<len(ground_truth_path[0,:])-skip-2:
                if nominal:
                    p1.remove()

                if linear_fb:
                    p2.remove()

                if ground_truth:
                    p3.remove()

                if show_ref:
                    p4.remove()
            
        

if make_vid:
    writer.finish()

if make_plot:
    reds=cm.get_cmap(name='Reds', lut=None)
    blues=cm.get_cmap(name='Blues', lut=None)
    greens=cm.get_cmap(name='Greens', lut=None)
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(ground_truth_path[0,:],ground_truth_path[4,:],ground_truth_path[8,:],c=blues(0.75))
        
    ax.plot(learned_path[0,:],learned_path[4,:],learned_path[8,:],c=greens(0.75))
        
    ax.plot(nominal_path[0,:],nominal_path[4,:],nominal_path[8,:],c=reds(0.75))

    ax.scatter(ground_truth_path[0,-1],ground_truth_path[4,-1],ground_truth_path[8,-1],c=[blues(0.75)],s=100,marker='x',label='True System')

    ax.scatter(learned_path[0,-1],learned_path[4,-1],learned_path[8,-1],c=[greens(0.75)],s=100,marker='x', label='Learned policy')

    ax.scatter(nominal_path[0,-1],nominal_path[4,-1],nominal_path[8,-1],c=[reds(0.75)],s=100,marker='x',label='Prior Model')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.9))
    ax.figure.savefig(output_dir + '3dpathsummary.png')








