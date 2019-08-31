from __future__ import division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import gym
import time
import spinup2.algos.vpg.core as core
from spinup2.utils.logx import EpochLogger
from spinup2.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup2.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from itertools import izip

import rospy
from quads_msgs.msg import LearnedParameters, Parameters

class VPGBuffer(object):
    u"""
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        u"""
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        u"""
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        u"""
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


u"""

Vanilla Policy Gradient

(with GAE-Lambda for advantage estimation)

"""

def vpgpolynomial(env_fn, actor_critic=core.polynomial_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.9, pi_lr=2e-5,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10, l1_scaling = 0.001):
    u"""

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    #ros stuff
    name = rospy.get_name() + "/ppo_rl_agent"
    params_topic = rospy.get_param("~topics/params")
    params_pub = rospy.Publisher(params_topic, LearnedParameters)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs[u'action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [v, logp]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in [u'pi', u'v'])
    logger.log(u'\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # VPG objectives
    var = [v for v in tf.trainable_variables() if u"pi" in v.name][ 0 ]
    norm_loss = l1_scaling*tf.norm(var,1)
    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute

    # Optimizers
    pi_optim = MpiAdamOptimizer(learning_rate=pi_lr)
    train_pi = pi_optim.minimize(pi_loss)
    train_pi_norm = pi_optim.minimize(norm_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={u'x': x_ph}, outputs={u'pi': pi, u'v': v})

    def update():
        inputs = dict((k, v) for k,v in izip(all_phs, buf.get()))
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # print "grads"
        # print sess.run(pi_optim.compute_gradients(pi_loss, tf.trainable_variables(u'pi')),
        #                feed_dict=inputs)

        # Policy gradient step
        sess.run(train_pi, feed_dict=inputs)
        sess.run(train_pi_norm, feed_dict=inputs)

        #polynomial penalizing number of terms
        # with tf.variable_scope('pi'):
        #     grads_and_vars = pi_optim.compute_gradients(tf.norm(tf.trainable_variables(),ord=1))
        #     pi_optim.apply_gradient(grads_and_vars)

        # Value function learning
        for _ in xrange(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, pi_l_norm = sess.run([pi_loss, v_loss, approx_kl,norm_loss], feed_dict=inputs)
        logger.store(LossNorm = pi_l_norm,LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

        # Publish ros parameters
        params_msg = LearnedParameters()
        params = [sess.run(v).flatten() for v in tf.trainable_variables() if u"pi" in v.name]
        num_params_in_msg = sum([len(p) for p in params])
        assert(num_params_in_msg == core.count_vars(u'pi'))
        for p in params:
            msg = Parameters()
            if isinstance(p, np.ndarray):
                msg.params = list(p)
            else:
                msg.params = [p]
            params_msg.params.append(msg)
        params_pub.publish(params_msg)

    start_time = time.time()
    #o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    env.reset()
    ep_ret = 0
    ep_len = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in xrange(epochs):
        for t in xrange(local_steps_per_epoch):
            o, r, a, d, _ = env.step()
            ep_ret += r
            ep_len += 1

            v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1), a_ph: a.reshape(1, -1)})

            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)


            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print u'Warning: trajectory cut off by epoch at %d steps.'%ep_len
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                _, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({u'env': env}, None)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular(u'Epoch', epoch)
        logger.log_tabular(u'EpRet', with_min_and_max=True)
        logger.log_tabular(u'EpLen', average_only=True)
        logger.log_tabular(u'VVals', with_min_and_max=True)
        logger.log_tabular(u'TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular(u'LossPi', average_only=True)
        logger.log_tabular(u'LossV', average_only=True)
        logger.log_tabular(u'DeltaLossPi', average_only=True)
        logger.log_tabular(u'DeltaLossV', average_only=True)
        logger.log_tabular(u'Entropy', average_only=True)
        logger.log_tabular(u'KL', average_only=True)
        logger.log_tabular(u'Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == u'__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(u'--env', type=unicode, default=u'HalfCheetah-v2')
    parser.add_argument(u'--hid', type=int, default=64)
    parser.add_argument(u'--l', type=int, default=2)
    parser.add_argument(u'--gamma', type=float, default=0.99)
    parser.add_argument(u'--seed', u'-s', type=int, default=0)
    parser.add_argument(u'--cpu', type=int, default=4)
    parser.add_argument(u'--steps', type=int, default=4000)
    parser.add_argument(u'--epochs', type=int, default=50)
    parser.add_argument(u'--exp_name', type=unicode, default=u'vpg')
    parser.add_argument(u'--order', type=int, default=2)
    parser.add_argument(u'--l1scaling', type=float, default=0.001)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    #polynomial version
    vpgpolynomial(lambda : gym.make(args.env), actor_critic=core.polynomial_actor_critic,
        ac_kwargs=dict(order=args.order), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, l1_scaling = args.l1_scaling)
