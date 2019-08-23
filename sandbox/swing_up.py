import numpy as np
#import control
from scipy.linalg import solve_continuous_are
from double_pendulum import DoublePendulum
import tensorflow as tf
import spinup
import matplotlib.pyplot as plt
from plotter import Plotter

#should be in form ./logs/ and then whatever foldername logger dumped
filename = 'ppo-dptest-uscaling1'


filepath="./logs/{}/simple_save".format(filename)

#load tensorflow graph from path
sess = tf.Session()
model = spinup.utils.logx.restore_tf_graph(sess,filepath)
print(model)


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

#    K,S,E=control.lqr(A,B,Q,R)

    return K



linear_fb=1
nominal=1
truedyn=1
T=500
to_render=1
check_energy=0
speed=0.001
path=0

# Create a double pendulum. And a bad one
mass1 = 1.0
mass2 = 1.0
length1 = 1.0
length2 = 1.0
time_step = 0.01
friction_coeff=0.5
dyn = DoublePendulum(mass1, mass2, length1, length2, time_step, friction_coeff)
bad_dyn = DoublePendulum(0.33*mass1, 0.33*mass2, 0.33*length1, 0.33*length2, time_step,friction_coeff)


# LQR Parameters and dynamics
q=200.0
r=1.0
A,B=get_Linear_System()
Q=q*np.diag([1,0,1.0,0])
R=r*np.eye(2)

#Get Linear Feedback policies
K=solve_lqr(A,B,Q,R)

#Get random initial state



reference=np.zeros((4,T))
reference[0,:]=np.pi #*np.sin(np.linspace(0,T*time_step,T))
reference[2,:]=np.pi #*np.cos(np.linspace(0,T*time_step,T))

learned_path=np.zeros((4,T+1))
learned_controls_path=np.zeros((2,T))

nominal_path=np.zeros((4,T+1))
nominal_controls_path=np.zeros((2,T))
true_path=np.zeros((4,T+1))
true_controls_path=np.zeros((2,T))
x0=0.0*np.ones((4,1))

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
        #desired_linear_system_state=dyn.wrap_angles(desired_linear_system_state)
        ref=np.reshape(reference[:,t],(4,1))
        #np.concatenate([y,ydot],axis=0)

        diff=dyn.observation_delta(ref,desired_linear_system_state)
        # diff=np.zeros((4,1))
        # ref1 = ref[0]
        # des1 = desired_linear_system_state[0]

        # ref2 = ref[2]
        # des2 = desired_linear_system_state[2]

        # term1=(des1 - ref1 + np.pi) % (2.0 * np.pi) - np.pi
        # term2=(ref1- des1 + np.pi) % (2.0 * np.pi) - np.pi
        # term3=(des2 - ref2 + np.pi) % (2.0 * np.pi) - np.pi
        # term4=(ref2 - des2 + np.pi) % (2.0 * np.pi) - np.pi

        # if abs(term1)<abs(term2):
        #     diff[0,0]= term1
        # else:
        #     diff[0,0]= -term2

        # if abs(term3)<abs(term4):
        #     diff[2,0]= term3
        # else:
        #     diff[2,0]= -term4

        # diff[1,0]=desired_linear_system_state[1]-ref[1]
        # diff[3,0]=desired_linear_system_state[3]-ref[3]

        v=-1*K @ (diff)

        obs = np.array(dyn.preprocess_state(x))
        # obs = x
    
        u = sess.run(tf.get_default_graph().get_tensor_by_name('pi/add:0'),{tf.get_default_graph().get_tensor_by_name('Placeholder:0'): obs.reshape(1,-1)})
         #output of neural network
        u = u[0]
        m2, f2 = np.split(u,[4])

        M = bad_dyn._M_q(x) + np.reshape(m2,(2, 2))

        f = bad_dyn._f_q(x) + np.reshape(f2,(2, 1))

        control = np.matmul(M, v) + f
        
        learned_controls_path[:,t]=control[:,0]
        x=dyn.integrate(x,control)
        learned_path[:,t+1]=(diff)[:,0]
        if check_energy:
            print(dyn.total_energy(x))
        if to_render and t%2==0:
            dyn.render(x,speed,path,float(t+0.2*T)/(1.5*T))

if nominal:

    x=x0.copy()
    dyn.fig=None

    for t in range(T):
        y=dyn.observation(x)
        ydot=dyn.observation_dot(x)

        desired_linear_system_state=np.zeros((4,1))
        desired_linear_system_state[0,0]=y[0,0]
        desired_linear_system_state[1,0]=ydot[0,0]
        desired_linear_system_state[2,0]=y[1,0]
        desired_linear_system_state[3,0]=ydot[1,0]
        #desired_linear_system_state=dyn.wrap_angles(desired_linear_system_state)
        ref=np.reshape(reference[:,t],(4,1))

        diff=np.zeros((4,1))
        ref1 = ref[0]
        des1 = desired_linear_system_state[0]

        ref2 = ref[2]
        des2 = desired_linear_system_state[2]

        term1=(des1 - ref1 + np.pi) % (2.0 * np.pi) - np.pi
        term2=(ref1- des1 + np.pi) % (2.0 * np.pi) - np.pi
        term3=(des2 - ref2 + np.pi) % (2.0 * np.pi) - np.pi
        term4=(ref2 - des2 + np.pi) % (2.0 * np.pi) - np.pi

        if abs(term1)<abs(term2):
            diff[0,0]= term1
        else:
            diff[0,0]= -term2

        if abs(term3)<abs(term4):
            diff[2,0]= term3
        else:
            diff[2,0]= -term4

        diff[1,0]=desired_linear_system_state[1]-ref[1]
        diff[3,0]=desired_linear_system_state[3]-ref[3]

        v=-1*K @ (diff)


        #v=-1*K @ (dyn.wrap_angles(desired_linear_system_state)-dyn.wrap_angles(ref))
        control=bad_dyn._M_q(x) @ v +bad_dyn._f_q(x)
        nominal_controls_path[:,t]=control[:,0]
        x=dyn.integrate(x,control)
        nominal_path[:,t+1]=(desired_linear_system_state)[:,0]
        if check_energy:
            print(dyn.total_energy(x))
        if to_render and t%2==0:
            dyn.render(x,speed,path,float(t+0.2*T)/(1.5*T))
if truedyn:

    x=x0.copy()
    dyn.fig=None

    for t in range(T):
        y=dyn.observation(x)
        ydot=dyn.observation_dot(x)

        desired_linear_system_state=np.zeros((4,1))
        desired_linear_system_state[0,0]=y[0,0]
        desired_linear_system_state[1,0]=ydot[0,0]
        desired_linear_system_state[2,0]=y[1,0]
        desired_linear_system_state[3,0]=ydot[1,0]
        #desired_linear_system_state=dyn.wrap_angles(desired_linear_system_state)
        ref=np.reshape(reference[:,t],(4,1))

        diff=np.zeros((4,1))
        ref1 = ref[0]
        des1 = desired_linear_system_state[0]

        ref2 = ref[2]
        des2 = desired_linear_system_state[2]

        term1=(des1 - ref1 + np.pi) % (2.0 * np.pi) - np.pi
        term2=(ref1- des1 + np.pi) % (2.0 * np.pi) - np.pi
        term3=(des2 - ref2 + np.pi) % (2.0 * np.pi) - np.pi
        term4=(ref2 - des2 + np.pi) % (2.0 * np.pi) - np.pi

        if abs(term1)<abs(term2):
            diff[0,0]= term1
        else:
            diff[0,0]= -term2

        if abs(term3)<abs(term4):
            diff[2,0]= term3
        else:
            diff[2,0]= -term4

        diff[1,0]=desired_linear_system_state[1]-ref[1]
        diff[3,0]=desired_linear_system_state[3]-ref[3]

        v=-1*K @ (diff)


        #v=-1*K @ (dyn.wrap_angles(desired_linear_system_state)-dyn.wrap_angles(ref))
        control=dyn._M_q(x) @ v +dyn._f_q(x)
        true_controls_path[:,t]=control[:,0]
        x=dyn.integrate(x,control)
        true_path[:,t+1]=(desired_linear_system_state)[:,0]
        if check_energy:
            print(dyn.total_energy(x))
        if to_render and t%2==0:
            dyn.render(x,speed,path,float(t+0.2*T)/(1.5*T))

#    plt.plot(nominal_path[0,:],'r')
#    plt.plot(nominal_path[2,:],'r')
#    plt.plot(reference[0,:],'b')
#    plt.plot(reference[2,:],'b')
#    plt.show()
