import numpy as np
from scipy.linalg import solve_continuous_are
import spinup2.algos.ppo.core as core
from gym import spaces
import tensorflow as tf
import sys
import rospy
from quads_msgs.msg import AuxiliaryControl
from quads_msgs.msg import OutputDerivatives
from quads_msgs.msg import Control
from quads_msgs.msg import State
from quads_msgs.msg import StateTransition
from quads_msgs.msg import LearnedParameters

from quadrotor_14d import Quadrotor14D

class FeedbackLinearizingController(object):
    def __init__(self):
        self._name = rospy.get_name() + "/feedback_linearizing_controller"

        if not self.load_parameters(): sys.exit(1)
        if not self.register_callbacks(): sys.exit(1)

        self._M1, self._f1 = self._dynamics.feedback_linearize()
        # self._params, self._M2, self._f2 = self._construct_learned_parameters()

        # LQR.
        self._A, self._B, _ = self._dynamics.linearized_system()
        Q = 1.0e-2 * np.diag([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        R = 1.0 * np.eye(4)

        def solve_lqr(A, B, Q, R):
            P = solve_continuous_are(A, B, Q, R)
            K = np.dot(np.dot(np.linalg.inv(R), B.T), P)
            return K

        self._K = solve_lqr(self._A, self._B, Q, R)

        # Initial reference is just a hover, but will be overwritten by msgs as we receive them.
        self._ref = np.zeros((14, 1))
        self._ref[8, 0] = 1.0

        # Output derivates are none for now.
        self._y = None

        # Linear system state.
        self._ylin = None

        #define placeholders
        observation_space = spaces.Box(low=-100,high=100,shape=(13,),dtype=np.float32)
        action_space = spaces.Box(low=-50,high=50,shape=(20,),dtype=np.float32)
        self._x_ph, self._u_ph = core.placeholders_from_spaces(observation_space, action_space)

        #define actor critic
        #TODO add in central way to accept arguments
        pi, logp, logp_pi, v = core.mlp_actor_critic(self._x_ph, self._u_ph)

        #start up tensorflow graph
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def load_parameters(self):
        if not rospy.has_param("~topics/y"):
            return False
        self._output_derivs_topic = rospy.get_param("~topics/y")

        if not rospy.has_param("~topics/x"):
            return False
        self._state_topic = rospy.get_param("~topics/x")

        if not rospy.has_param("~topics/params"):
            return False
        self._params_topic = rospy.get_param("~topics/params")

        if not rospy.has_param("~topics/u"):
            return False
        self._control_topic = rospy.get_param("~topics/u")

        if not rospy.has_param("~topics/ref"):
            return False
        self._ref_topic = rospy.get_param("~topics/ref")

        if not rospy.has_param("~dynamics/m"):
            return False
        m = rospy.get_param("~dynamics/m")

        if not rospy.has_param("~dynamics/Ix"):
            return False
        Ix = rospy.get_param("~dynamics/Ix")

        if not rospy.has_param("~dynamics/Iy"):
            return False
        Iy = rospy.get_param("~dynamics/Iy")

        if not rospy.has_param("~dynamics/Iz"):
            return False
        Iz = rospy.get_param("~dynamics/Iz")

        self._dynamics = Quadrotor14D(m, Ix, Iy, Iz)

        return True

    def register_callbacks(self):
        self._params_sub = rospy.Subscriber(
            self._params_topic, LearnedParameters, self.params_callback)

        self._state_sub = rospy.Subscriber(
            self._state_topic, State, self.state_callback)

        self._output_derivs_sub = rospy.Subscriber(
            self._output_derivs_topic, OutputDerivatives, self.output_callback)

        self._ref_sub = rospy.Subscriber(
            self._ref_topic, OutputDerivatives, self.ref_callback)

        self._control_pub = rospy.Publisher(self._control_topic, Control)

        return True

    def params_callback(self, msg):
        # TODO(@shreyas, @eric): Update values of self._params here.
        # change network parameters
        #msg probably needs to be formatted
        tf.assign(msg,tf.trainable_variables())

    def ref_callback(self, msg):
        self._ref[0, 0] = msg.x
        self._ref[1, 0] = msg.xdot1
        self._ref[2, 0] = msg.xdot2
        self._ref[3, 0] = msg.xdot3
        self._ref[4, 0] = msg.y
        self._ref[5, 0] = msg.ydot1
        self._ref[6, 0] = msg.ydot2
        self._ref[7, 0] = msg.ydot3
        self._ref[8, 0] = msg.z
        self._ref[9, 0] = msg.zdot1
        self._ref[10, 0] = msg.zdot2
        self._ref[11, 0] = msg.zdot3
        self._ref[12, 0] = msg.psi
        self._ref[13, 0] = msg.psidot1

    def state_callback(self, msg):
        # Update x.
        x = np.array([[msg.x], [msg.y], [msg.z], [msg.theta],
                      [msg.phi], [msg.psi], [msg.dx], [msg.dy],
                      [msg.dz], [msg.zeta], [msg.xi], [msg.q], [msg.r], [msg.p]])

        # Determine v.
        if self._y is not None:
            v = -np.dot(self._K, (self._y - self._ref))
            u = self.feedback(x, v)

            # Publish Control msg.
            u_msg = Control()
            u_msg.thrustdot2 = u[0, 0]
            u_msg.pitchdot2 = u[1, 0]
            u_msg.rolldot2 = u[2, 0]
            u_msg.yawdot2 = u[3, 0]
            self._control_pub.publish(u_msg)

            # Publish StateTransition msg.
            t_msg = StateTransition()
            t_msg.x = x.flatten()
            t_msg.u = u.flatten()
            t_msg.v = v.flatten()
            t_msg.r = -self._dynamics.observation_distance(self._y, self._ylin, norm=2)

    def output_callback(self, msg):
        self._y = np.array([[msg.x], [msg.xdot1], [msg.xdot2], [msg.xdot3],
                            [msg.y], [msg.ydot1], [msg.ydot2], [msg.ydot3],
                            [msg.z], [msg.zdot1], [msg.zdot2], [msg.zdot3],
                            [msg.psi], [msg.psidot1]])

        # Handle no linear system state case.
        if self._ylin is None:
            self._ylin = self._y
            self._last_ylin_reset_time = rospy.Time.now().to_sec()
            self._last_ylin_integration_time = self._last_ylin_reset_time
        elif rospy.Time.now().to_sec() - self._last_ylin_reset_time > 0.5:
            # Been too long. Reset.
            self._ylin = self._y
            self._last_ylin_integration_time = rospy.Time.now().to_sec()
        else:
            # Integrate forward.
            dt = rospy.Time.now().to_sec() - self._last_ylin_integration_time
            self._last_ylin_integration_time = rospy.Time.now().to_sec()

            v = -np.dot(self._K, (self._ylin - self._ref))
            self._ylin += dt * (np.dot(self._A, self._ylin) + np.dot(self._B, v))

    def feedback(self, x, v):
        """ Compute u from x, v (np.arrays). See above comment for details. """
        v = np.reshape(v, (4, 1))
        x = self.preprocess_state(x)
        a = self._sess.run(self._pi, feed_dict={self._x_ph: v.reshape(1,-1)})

        #creating m2, ft
        m2, f2 = np.split(self._uscaling * u,[16])

        # TODO: make sure this works with tf stuff.
        return np.dot(self._M1(x) + m2, v) + f2 + self.f2(x)

    # def _construct_learned_parameters(self):
    #     """ Create params, M2, f2. """
    #     pass
    def preprocess_state(self, x):
        x[0] = np.sin(x[3])
        x[1] = np.sin(x[4])
        x[2]= np.sin(x[5])
        x[3] = np.cos(x[3])
        x[4] = np.cos(x[4])
        x[5]= np.cos(x[5])
        x.pop(10)
        return x
    
