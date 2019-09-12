#!/usr/bin/env python


# Python Imports
import sys
import numpy as np
import itertools
from scipy.linalg import solve_continuous_are, expm
from gym import spaces
import tensorflow as tf

# ROS imports
import rospy
import baxter_interface
import moveit_commander
from baxter_pykdl import baxter_kinematics
from baxter_learning_msgs.msg import State, Reference, DataLog
from quads_msgs.msg import Transition
from quads_msgs.msg import LearnedParameters
from std_msgs.msg import Empty

# Package Imports
import utils
from path_planner import PathPlanner

# import spinup2.algos.vpg.core as core
import spinup2.algos.ppo.core as core



class BaxterLearning():
    """docstring for BaxterLearning"""
    def __init__(self):

        if not self.load_parameters(): sys.exit(1)

        self._limb = baxter_interface.Limb(self._arm)
        self._kin = baxter_kinematics(self._arm)
        self._planner = PathPlanner('{}_arm'.format(self._arm))

        rospy.on_shutdown(self.shutdown)

        plan = self._planner.plan_to_joint_pos(np.array([-0.6, -0.4, -0.5, 0.6, -0.4, 1.1, -0.5]))
        self._planner.execute_plan(plan)
        rospy.sleep(5)

        ################################## Tensorflow bullshit
        if self._learning_bool:
            # I DON'T KNOW HOW THESE WORK
            #define placeholders
            observation_space = spaces.Box(low=-100,high=100,shape=(21,),dtype=np.float32)
            action_space = spaces.Box(low=-50,high=50,shape=(56,),dtype=np.float32)
            self._x_ph, self._u_ph = core.placeholders_from_spaces(observation_space, action_space)

            #define actor critic
            #TODO add in central way to accept arguments
            self._pi, self._logp, self._logp_pi, self._v = core.mlp_actor_critic(
               self._x_ph, self._u_ph, hidden_sizes=(64,2), action_space=action_space)
            # POLY_ORDER = 2
            # self._pi, self._logp, self._logp_pi, self._v = core.polynomial_actor_critic(
            #     self._x_ph, self._u_ph, POLY_ORDER, action_space=action_space)

            #start up tensorflow graph

            var_counts = tuple(core.count_vars(scope) for scope in [u'pi'])
            print(u'\nYoyoyoyyoyo Number of parameters: \t pi: %d\n'%var_counts)

            self._tf_vars = [v for v in tf.trainable_variables() if u'pi' in v.name]
            self._num_tf_vars = sum([np.prod(v.shape.as_list()) for v in self._tf_vars])
            print("tf vars is of length: ", len(self._tf_vars))
            print("total trainable vars: ", len(tf.trainable_variables()))

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())

            print("total trainable vars: ", len(tf.trainable_variables()))

        self._last_time = rospy.Time.now().to_sec()

        current_position = utils.get_joint_positions(self._limb).reshape((7,1))
        current_velocity = utils.get_joint_velocities(self._limb).reshape((7,1))

        self._last_x = np.vstack([current_position, current_velocity])

        if self._learning_bool:
            self._last_a = self._sess.run(self._pi, feed_dict={self._x_ph: self.preprocess_state(self._last_x).reshape(1,-1)})


        ##################################### Controller params

        self._A = np.vstack([
            np.hstack([np.zeros((7,7)), np.eye(7)]),
            np.hstack([np.zeros((7,7)), np.zeros((7,7))])
            ])
        self._B = np.vstack([np.zeros((7,7)), np.eye(7)])
        Q = 0.2*np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        R = np.eye(7)

        def solve_lqr(A, B, Q, R):
            P = solve_continuous_are(A, B, Q, R)
            K = np.dot(np.dot(np.linalg.inv(R), B.T), P)
            return K

        self._K = solve_lqr(self._A, self._B, Q, R)

        # self._Kp = 6*np.diag(np.array([1, 1, 1.5, 1.5, 1, 1, 1]))
        # self._Kv = 5*np.diag(np.array([2, 2, 1, 1, 0.8, 0.3, 0.3]))

        # self._Kp = 9*np.diag(np.array([4, 6, 4, 8, 1, 5, 1]))
        # self._Kv = 5*np.diag(np.array([2, 3, 2, 4, 0.8, 1.5, 0.3]))

        self._Kp = 6*np.diag(np.array([1, 1, 1, 1, 1, 1, 1]))
        self._Kv = 6*np.diag(np.array([1, 1, 1, 1, 1, 1, 1]))

        if not self.register_callbacks(): sys.exit(1)


    def load_parameters(self):
        if not rospy.has_param("~baxter/arm"):
            return False
        self._arm = rospy.get_param("~baxter/arm")

        if not rospy.has_param("~learning"):
            return False
        self._learning_bool = rospy.get_param("~learning")

        if not rospy.has_param("~topics/ref"):
            return False
        self._ref_topic = rospy.get_param("~topics/ref")

        if not rospy.has_param("~topics/params"):
            return False
        self._params_topic = rospy.get_param("~topics/params")

        if not rospy.has_param("~topics/transitions"):
            return False
        self._transitions_topic = rospy.get_param("~topics/transitions")

        if not rospy.has_param("~topics/linear_system_reset"):
            return False
        self._reset_topic = rospy.get_param("~topics/linear_system_reset")

        if not rospy.has_param("~topics/integration_reset"):
            return False
        self._int_reset_topic = rospy.get_param("~topics/integration_reset")

        if not rospy.has_param("~topics/data"):
            return False
        self._data_topic = rospy.get_param("~topics/data")

        return True

    def register_callbacks(self):

        self._params_sub = rospy.Subscriber(
            self._params_topic, LearnedParameters, self.params_callback)

        self._ref_sub = rospy.Subscriber(
            self._ref_topic, Reference, self.ref_callback)

        self._reset_sub = rospy.Subscriber(
            self._reset_topic, Empty, self.linear_system_reset_callback)

        self._int_reset_sub = rospy.Subscriber(
            self._int_reset_topic, Empty, self.integration_reset_callback)

        self._transitions_pub = rospy.Publisher(self._transitions_topic, Transition)

        self._data_pub = rospy.Publisher(self._data_topic, DataLog)

        return True

    def params_callback(self, msg):
        t = rospy.Time.now().to_sec()
        sq_norm_old_params = 0.0
        sq_norm_difference = 0.0
        norm_params = []
        num_params = 0
        for p, v in zip(msg.params, self._tf_vars):
            num_params += len(p.params)
            print("P is len ", len(p.params), ", and v is len ", np.prod(v.shape.as_list()))
            assert(len(p.params) == np.prod(v.shape.as_list()))

            reshaped_p = np.array(p.params).reshape(v.shape.as_list())
            norm_params.append(np.linalg.norm(reshaped_p))
            v_actual = self._sess.run(v)
            sq_norm_old_params += np.linalg.norm(v_actual)**2
            sq_norm_difference += np.linalg.norm(v_actual - np.array(reshaped_p))**2
            self._sess.run(tf.assign(v, reshaped_p))

        rospy.loginfo("Updated tf params in controller.")
        rospy.loginfo("Parameters changed by %f on average." % (
            sq_norm_difference / float(num_params)))
        rospy.loginfo("Parameter group norms: " + str(norm_params))
        rospy.loginfo("Params callback took %f seconds." % (rospy.Time.now().to_sec() - t))

    def ref_callback(self, msg):

        # Get/log state data
        ref = msg

        dt = rospy.Time.now().to_sec() - self._last_time
        self._last_time = rospy.Time.now().to_sec()

        current_position = utils.get_joint_positions(self._limb).reshape((7,1))
        current_velocity = utils.get_joint_velocities(self._limb).reshape((7,1))

        current_state = State(current_position, current_velocity)

        # get dynamics info

        positions_dict = utils.joint_array_to_dict(current_position, self._limb)
        velocity_dict = utils.joint_array_to_dict(current_velocity, self._limb)

        inertia = self._kin.inertia(positions_dict)
        coriolis = self._kin.coriolis(positions_dict, velocity_dict)[0][0]
        coriolis = np.array([float(coriolis[i]) for i in range(7)]).reshape((7,1))

        gravity_wrench = np.array([0,0,0.981,0,0,0]).reshape((6,1))
        gravity_jointspace = (np.matmul(self._kin.jacobian_transpose(positions_dict), gravity_wrench))
        gravity = (np.matmul(inertia, gravity_jointspace)).reshape((7,1))

        # Linear Control

        error = np.array(ref.setpoint.position).reshape((7,1)) - current_position
        d_error = np.array(ref.setpoint.velocity).reshape((7,1)) - current_velocity
        # d_error = np.zeros((7,1))
        error_stack = np.vstack([error, d_error])

        v = np.matmul(self._Kv, d_error).reshape((7,1)) + np.matmul(self._Kp, error).reshape((7,1)) + np.array(ref.feed_forward).reshape((7,1))
        # v = np.array(ref.feed_forward).reshape((7,1))
        # v = np.matmul(self._K, error_stack).reshape((7,1))
        # v = np.zeros((7,1))

        # print current_position

        ##### Nonlinear control

        if self._learning_bool:
            x = np.vstack([current_position, current_velocity])
            a = self._sess.run(self._pi, feed_dict={self._x_ph: self.preprocess_state(x).reshape(1,-1)})
            m2, f2 = np.split(0.1*a[0],[49])
            m2 = m2.reshape((7,7))
            f2 = f2.reshape((7,1))

            x_predict = np.matmul(expm((self._A + np.matmul(self._B, np.hstack([self._Kp, self._Kv])))*dt), self._last_x)


            t_msg = Transition()
            t_msg.x = list(self._last_x.flatten())
            t_msg.a = list(self._last_a.flatten())
            t_msg.r = -np.linalg.norm(x - x_predict, 2)
            self._transitions_pub.publish(t_msg)
            data = DataLog(current_state, ref.setpoint, t_msg)

            self._last_x = x
            self._last_a = a
        else:
            m2 = np.zeros((7,7))
            f2 = np.zeros((7,1))
            data = DataLog(current_state, ref.setpoint, Transition())

        self._data_pub.publish(data)

        torque_lim = np.array([4, 4, 4, 4, 2, 2, 2]).reshape((7,1))

        torque = np.matmul(inertia + m2, v) + coriolis + gravity + f2

        torque = np.clip(torque, -torque_lim, torque_lim).reshape((7,1))

        torque_dict = utils.joint_array_to_dict(torque, self._limb)
        self._limb.set_joint_torques(torque_dict)

    def linear_system_reset_callback(self, msg):
        # plan = self._planner.plan_to_joint_pos(np.zeros(7))
        # self._planner.execute_plan(plan)
        pass

    def integration_reset_callback(self, msg):
        self._last_time = rospy.Time.now().to_sec()

    def preprocess_state(self, x0):
        q = x0[0:7]
        dq = x0[7:14]
        x = np.hstack([np.sin(q), np.cos(q), dq])
        return x

    def shutdown(self):
        """
        Code to run on shutdown. This is good practice for safety
        """
        rospy.loginfo("Stopping Controller")

        # Set velocities to zero
        zero_vel_dict = utils.joint_array_to_dict(np.zeros(7), self._limb)
        self._limb.set_joint_velocities(zero_vel_dict)

        # self._planner.stop()

        rospy.sleep(0.1)



if __name__ == '__main__':

    rospy.init_node("baxter_learning")

    bl = BaxterLearning()

    rospy.spin()
