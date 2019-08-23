import numpy as np
from scipy.linalg import solve_continuous_are
from diff_drive import DiffDrive
import tensorflow as tf
import spinup
import matplotlib.pyplot as plt


#should be in form ./logs/ and then whatever foldername logger dumped
filename = 'ppo-diffdrivetest1'

filepath="./logs/{}/simple_save".format(filename)

#load tensorflow graph from path
sess = tf.Session()
model = spinup.utils.logx.restore_tf_graph(sess,filepath)
print(model)


def solve_lqr(A,B,Q,R):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


learned = 1
truedyn=1
T=500
to_render=1


# Create a double pendulum. And a bad one
dyn = DiffDrive()

# LQR Parameters and dynamics
q=1000.0
r=1.0
A,B, C = dyn.linearized_system()
Q=q*np.diag([1,1,1,1])
R=r*np.eye(2)
#Get Linear Feedback policies
K=solve_lqr(A,B,Q,R)

#set reference
reference=np.zeros((4,T))
reference[0,:]=5
reference[1,:]=5
reference[2,:]=5
reference[3,:]=0

learned_path=np.zeros((4,T))
ground_truth_path = np.zeros((4,T))
x0=np.zeros((4,1))
x0[0,0] = 0
x0[1,0] = 0
x0[2,0] = np.pi/2
x0[3,0] = 20

if learned:
    x=x0.copy()
    y = dyn.linearized_system_state(x)
    for t in range(T):
        #compute linear controls
        diff = y - reference[:,t]
        v=-1*K @ (diff)

        #preprocess state
        obs = np.array(dyn.preprocess_state(x))

        #get nn output
        u = sess.run(tf.get_default_graph().get_tensor_by_name('pi/add:0'),{tf.get_default_graph().get_tensor_by_name('Placeholder:0'): obs.reshape(1,-1)})
        u = u[0]

        #transform controls
        m2, f2 = np.split(u,[4])
        M = np.reshape(m2,(2, 2))
        f = np.reshape(f2,(2, 1))
        control = np.matmul(M, v) + f


        #integrate forward
        x = dyn.integrate(x, control, 0.01)
        y = dyn.linearized_system_state(x)

        #store data
        learned_path[:,t] = y.copy().flatten()

if truedyn:
    x=x0.copy()
    y = dyn.linearized_system_state(x)
    M1, f1 = dyn.feedback_linearize()
    for t in range(T):
        #compute linear controls
        diff = y - reference[:,t]
        v=-1*K @ (diff)

        #transform controls
        control = np.matmul(M1(x), v) + f1(x)

        #integrate forward
        x = dyn.integrate(x, control, 0.01)
        y = dyn.linearized_system_state(x)

        #store data
        ground_truth_path[:,t] = y.copy().flatten()

plt.figure()
plt.plot(learned_path[0,:], learned_path[1,:])
plt.plot(ground_truth_path[0,:], ground_truth_path[1,:])
plt.show()


        
       
