#!/usr/bin/env python

"""
Partially stolen from Berkeley EE106A lab 3 prelab, which was written by Aaron Bestick and edited by Amay Saxena
"""


import numpy as np
import scipy.linalg as spl


np.set_printoptions(precision=4,suppress=True)

def skew_3d(omega):
    """
    Converts a rotation vector in 3D to its corresponding skew-symmetric matrix.
    
    Args:
    omega - (3,) ndarray: the rotation vector
    
    Returns:
    omega_hat - (3,3) ndarray: the corresponding skew symmetric matrix
    """
    if not omega.shape == (3,):
        raise TypeError('omega must be a 3-vector')

    return np.array([[0, -omega[2], omega[1]],
                    [omega[2], 0, -omega[0]],
                    [-omega[1], omega[0], 0]])

def unskew_3d(omega_hat):
    """
    Converts a skew-symmetric matrix to a rotation vector in R3

    Args:
    omega_hat - (3,3) ndarray: the corresponding skew symmetric matrix
    
    Returns:
    omega - (3,) ndarray: the rotation vector

    """

    if not omega_hat.shape == (3,3):
        raise TypeError('omega_hat must be a 3x3 matrix')

    return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0] ])


def rotation_3d(omega, theta):
    """
    Computes a 3D rotation matrix given a rotation axis and angle of rotation.
    
    Args:
    omega - (3,) ndarray: the axis of rotation
    theta: the angle of rotation
    
    Returns:
    rot - (3,3) ndarray: the resulting rotation matrix
    """
    if not omega.shape == (3,):
        raise TypeError('omega must be a 3-vector')
    
    hat_u = skew_3d(omega)
    theta = theta * np.linalg.norm(omega)
    hat_u = hat_u / np.linalg.norm(omega)
    return np.eye(3) + hat_u * np.sin(theta) + np.dot(hat_u, hat_u) * (1 - np.cos(theta))

def inverse_rotation_3d(R):
    """
    Computes omega and theta from R
    """

    theta = np.arccos((np.trace(R) - 1)/2)
    if np.abs(theta) > 1e9:
        omega = np.array([
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            ])
        omega = (1/(2*np.sin(theta)))*omega
    else:
        omega = np.zeros((3,))

    return omega*theta

def renormalize_rotation_3d(R):
    theta = np.arccos((np.trace(R) - 1)/2)
    if np.abs(theta) > 1e9:
        omega = np.array([
                R[2,1] - R[1,2],
                R[0,2] - R[2,0],
                R[1,0] - R[0,1]
            ])
        omega = (1/(2*np.sin(theta)))*omega

        q = np.concatenate(
            [np.array([cos(theta/2)]), omega*np.sin(theta/2)])
        q = q/np.linalg.norm(q,2)

        theta_norm = 2*np.arccos(q[0])
        omega_norm = q[1:]/np.sin(theta_norm/2)
        R_norm = rotation_3d(omega_norm, theta_norm)

    else: 
        return R

def hat_3d(xi):
    """
    Converts a 3D twist to its corresponding 4x4 matrix representation
    
    Args:
    xi - (6,) ndarray: the 3D twist
    
    Returns:
    xi_hat - (4,4) ndarray: the corresponding 4x4 matrix
    """
    if not xi.shape == (6,):
        raise TypeError('xi must be a 6-vector')

    v = xi[:3]
    w = xi[3:]
    xi_hat = np.zeros((4, 4))
    xi_hat[:3, :3] = skew_3d(w)
    xi_hat[:3, 3] = v
    return xi_hat

def unhat_3d(xi_hat):
    """
    Converts a 4x4 matrix in se(3) to the corresponding twist
    
    Args:
    xi_hat - (4,4) ndarray: the corresponding 4x4 matrix

    Returns:
    xi - (6,) ndarray: the 3D twist

    """

    v = xi_hat[:3,3]
    w = unskew_3d(xi_hat[:3, :3])

    return np.concatenate([v,w])

def homog_3d(xi, theta):
    """
    Computes a 4x4 homogeneous transformation matrix given a 3D twist and a 
    joint displacement.
    
    Args:
    xi - (6,) ndarray: the 3D twist
    theta: the joint displacement
    Returns:
    g - (4,4) ndarary: the resulting homogeneous transformation matrix
    """
    if not xi.shape == (6,):
        raise TypeError('xi must be a 6-vector')

    v = xi[:3]
    w = xi[3:]
    I = np.eye(3)
    if np.allclose(w, 0):
        # Pure translation
        R = I
        p = v * theta
    else:
        # Translation and rotation
        R = rotation_3d(w, theta)
        p = (1/np.square(np.linalg.norm(w))) * ((np.dot(np.dot((I - R), skew_3d(w)), v)) + theta * np.dot(np.outer(w, w), v))
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = p
    return g

def homog_3d_exp(xi):
    return spl.expm(self.hat_3d(xi))

def inverse_homog_3d(g):
    xi_hat = spl.logm(g)
    return unhat_3d(xi_hat)

def pR_to_g(p, R):
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = p
    return g

def g_to_pR(g):
    p = g[:3,3]
    R = g[:3,:3]
    return p, R

def renormalize_homog_3d(g):
    R = g[:3,:3]
    R = renormalize_rotation_3d(R)
    g[:3,:3] = R
    return g

def rbt_inv(g):
    R = g[:3,:3]
    p = g[:3, 3]
    g[:3,:3] = R.transpose()
    g[:3, 3] = - np.dot(R.transpose(), p)


def prod_exp(xi, theta):
    """
    Computes the product of exponentials for a kinematic chain, given 
    the twists and displacements for each joint.
    
    Args:
    xi - (6, N) ndarray: the twists for each joint
    theta - (N,) ndarray: the displacement of each joint
    
    Returns:
    g - (4,4) ndarray: the resulting homogeneous transformation matrix
    """
    if not xi.shape[0] == 6:
        raise TypeError('xi must be a Nx6')
    if not xi.shape[1] == theta.shape[0]:
        raise TypeError('there must be the same number of twists as joint angles.')

    g = np.eye(4)
    for i in range(xi.shape[1]):
        xi_i = xi[:, i]
        theta_i = theta[i]
        g_i = homog_3d(xi_i, theta_i)
        g = np.dot(g, g_i)
    return g



#-----------------------------Testing code--------------------------------------
#-------------(you shouldn't need to modify anything below here)----------------

def array_func_test(func_name, args, ret_desired):
    ret_value = func_name(*args)
    if not isinstance(ret_value, np.ndarray):
        print('[FAIL] ' + func_name.__name__ + '() returned something other than a NumPy ndarray')
    elif ret_value.shape != ret_desired.shape:
        print('[FAIL] ' + func_name.__name__ + '() returned an ndarray with incorrect dimensions')
    elif not np.allclose(ret_value, ret_desired, rtol=1e-3):
        print('[FAIL] ' + func_name.__name__ + '() returned an incorrect value')
    else:
        print('[PASS] ' + func_name.__name__ + '() returned the correct value!')

if __name__ == "__main__":
    print('Testing...')

    #Test skew_3d()
    arg1 = np.array([1.0, 2, 3])
    func_args = (arg1,)
    ret_desired = np.array([[ 0., -3.,  2.],
                            [ 3., -0., -1.],
                            [-2.,  1.,  0.]])
    array_func_test(skew_3d, func_args, ret_desired)

    #Test unskew_3d()
    array_func_test(unskew_3d, (ret_desired,), arg1)


    #Test rotation_3d()
    arg1 = np.array([2.0, 1, 3])
    arg2 = 0.587
    func_args = (arg1,arg2)
    ret_desired = np.array([[-0.1325, -0.4234,  0.8962],
                            [ 0.8765, -0.4723, -0.0935],
                            [ 0.4629,  0.7731,  0.4337]])
    array_func_test(rotation_3d, func_args, ret_desired)

    #Test renormalize_rotation_3d
    # print('Testing Renormalize. The first test may fail on value (this is ok)')
    func_args = (ret_desired+np.random.normal(0.0, 1e-5, (3,3)), )
    array_func_test(renormalize_rotation_3d, func_args, ret_desired)


    #Test hat_3d()
    arg1 = np.array([2.0, 1, 3, 5, 4, 2])
    func_args = (arg1,)
    ret_desired = np.array([[ 0., -2.,  4.,  2.],
                            [ 2., -0., -5.,  1.],
                            [-4.,  5.,  0.,  3.],
                            [ 0.,  0.,  0.,  0.]])
    array_func_test(hat_3d, func_args, ret_desired)

    #Test unhat_3d()
    array_func_test(unhat_3d, (ret_desired,), arg1)

    #Test homog_3d()
    arg1 = np.array([2.0, 1, 3, 5, 4, 2])
    arg2 = 0.658
    func_args = (arg1,arg2)
    ret_desired = np.array([[ 0.4249,  0.8601, -0.2824,  1.7814],
                            [ 0.2901,  0.1661,  0.9425,  0.9643],
                            [ 0.8575, -0.4824, -0.179 ,  0.1978],
                            [ 0.    ,  0.    ,  0.    ,  1.    ]])
    array_func_test(homog_3d, func_args, ret_desired)

    # Test inverse_homog_3d()
    xi = arg1/(np.linalg.norm(arg1[3:]))
    array_func_test(inverse_homog_3d, (spl.expm(hat_3d(xi)),), xi)

    #Test prod_exp()
    arg1 = np.array([[2.0, 1, 3, 5, 4, 6], [5, 3, 1, 1, 3, 2], [1, 3, 4, 5, 2, 4]]).T
    arg2 = np.array([0.658, 0.234, 1.345])
    func_args = (arg1,arg2)
    ret_desired = np.array([[ 0.4392,  0.4998,  0.7466,  7.6936],
                            [ 0.6599, -0.7434,  0.1095,  2.8849],
                            [ 0.6097,  0.4446, -0.6562,  3.3598],
                            [ 0.    ,  0.    ,  0.    ,  1.    ]])
    array_func_test(prod_exp, func_args, ret_desired)

    print('Done!')



