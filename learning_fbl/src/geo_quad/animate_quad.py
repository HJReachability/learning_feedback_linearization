import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from pytransform3d.plot_utils import Frame, Trajectory
import se3_utils as se3
import fbl_core.utils as utils

# For plotting quad
# from matplotlib import artist
# from mpl_toolkits.mplot3d import proj3d
# from mpl_toolkits.mplot3d.art3d import Line3D, Text3D, Poly3DCollection
# from pytransform3d.transformations import transform


class QuadAnimator(object):
    """docstring for QuadAnimator"""
    def __init__(self, low_lims, high_lims, dt=0.05):
        self.high_lims = high_lims
        self.low_lims = low_lims
        self.dt = dt

    def update_animation(self, state):
        # self.state_frame.remove()
        g = state2g(state)
        self.state_frame.set_data(g)

        self.xs.append(state[0])
        self.ys.append(state[1])
        self.zs.append(state[2])
        self.ax.plot3D(self.xs, self.ys, self.zs, 'black')

        plt.pause(self.dt)

    def plot_line(self, xs, ys, zs, color='black'):
        self.ax.plot3D(xs, ys, zs, color)


    def start_animation(self, state, refs, num_frames=10):
        """
        Starts the animation plot
        """

        self.refs = refs

        ref_frames = []
        xs = []
        ys = []
        zs = []
        rate = max([len(refs) // num_frames, 1])
        for index in range(len(refs)):
            # if index%rate == 0 or index==len(refs)-1:
            #     # import pdb 
            #     # pdb.set_trace()
            #     ref_frame = PartialFrame(refs[index])
            #     ref_frame.add_frame(self.ax)
            ref_frames.append(ref2g(refs[index]))
            xs.append(refs[index][0])
            ys.append(refs[index][1])
            zs.append(refs[index][2])

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        self.fig = plt.figure(figsize=(5,5))

        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.set_xlim((np.maximum(self.low_lims[0], np.min(xs)-2), np.minimum(self.high_lims[0], np.max(xs)+2) ) )
        self.ax.set_ylim((np.maximum(self.low_lims[1], np.min(ys)-2), np.minimum(self.high_lims[1], np.max(ys)+2) ) )
        self.ax.set_zlim((np.maximum(self.low_lims[2], np.min(zs)-2), np.minimum(self.high_lims[2], np.max(zs)+2) ) )
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ref_traj = Trajectory(np.array(ref_frames), n_frames=20, s=0.2)
        self.state_frame = state2frame(state, 1.0, 'Quad')
        self.start_frame = state2frame(state, 0.5, 'Start')
        self.end_frame = Frame(ref_frames[-1], 'End', 0.5)
        self.xs = [state[0]]
        self.ys = [state[1]]
        self.zs = [state[2]]

        self.ref_traj.add_trajectory(self.ax)
        self.state_frame.add_frame(self.ax)
        self.start_frame.add_frame(self.ax)
        self.end_frame.add_frame(self.ax)

        plt.pause(self.dt)
        # plt.show()

        return

def state2frame(x, s, label = None):
    """
    Given a quad state, make a Frame object
    x: state
    s: length of frame axes
    label: frame label
    """

    g = state2g(x)
    frame = Frame(g, label, s)

    return frame

def state2g(x):
    p = x[:3]
    R = np.reshape(x[3:12], (3,3))
    g = np.eye(4)
    g[:3,:3] = R
    g[:3,3] = p
    
    return g

def ref2g(r):
    p = r[:3]
    b1d = r[15:18]
    b3 = np.array([0,0,1])

    b2d = utils.unitify(np.cross(b3, b1d))
    b3d = utils.unitify(np.cross(b1d, b2d))
    Rd = np.array([b1d, b2d, b3d])

    g = np.eye(4)
    g[:3,:3] = Rd
    g[:3,3] = p
    return g

# class PartialFrame(Frame):
#     """Makes a partial frame from the reference trajectory"""
#     def __init__(self, r, ):
        

#         print(g)

#         super(PartialFrame, self).__init__(g, 0.5, None)
        







