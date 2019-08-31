from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    if(np.isscalar(shape)):
        return (length, shape)
    else:
        l = list(shape)
        l.insert(0, length)
        l = tuple(l)
        return l
#        tp = (length, shape)
#        return convert([element for tupl in tp for element in tupl])

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    x = tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
    return x

def get_vars(scope=u''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=u''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    u"""
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def nextOrderPolynomial(x, length, lastOrderPolynomial, incrementList):
    #empty list for next order
    nextOrderPolynomial = []

    for i in xrange(length):
        for j in xrange(incrementList[i]):
            nextOrderPolynomial.append(tf.multiply(x[:,i],lastOrderPolynomial[j]))

    return nextOrderPolynomial

def nextIncrementList(incrementList):
    nextIncrementList = [1]
    for i in xrange(len(incrementList)-1):
        nextIncrementList.append(nextIncrementList[i]+incrementList[i+1])
    return nextIncrementList


def polynomial(x, order, u_dim):
    u""" Computes u, a polynomial function of x (for each row of x).
    Polynomial will be composed of monomials of degree up to and including
    the specified 'order'.
    """
    NUM_STATE_DIMS = x.get_shape().as_list()[-1]

    print u"Generating monomials..."
    FullPolynomial = []
    z = tf.ones_like(x[:,0])
    FullPolynomial.append(z)
    incrementList = []
    lastPolynomial = []
    for i in xrange(NUM_STATE_DIMS):
        incrementList.append(i+1)
        lastPolynomial.append(x[:,i])


    #generate full polynomial
    FullPolynomial.extend(lastPolynomial)
    for i in xrange(order - 1):
        nextPolynomial = nextOrderPolynomial(x,NUM_STATE_DIMS,lastPolynomial,incrementList)
        FullPolynomial.extend(nextPolynomial)
        incrementList = nextIncrementList(incrementList)
        lastPolynomial = nextPolynomial
    print u"number of monomials: ", len(FullPolynomial)
    print u"...done!"


    # Now declare tf variables for the coefficients of all these monomials.
    coeffs = tf.get_variable(u"polynomial_coefficients",
                             initializer=tf.zeros((len(FullPolynomial), u_dim)),dtype=tf.float32)

    FullPolynomial = tf.transpose(tf.stack(FullPolynomial))

    # Compute polynomial output for each state.
    action = tf.matmul(FullPolynomial,coeffs)
    return action

u"""
Policies
"""


def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name=u'log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi

def polynomial_gaussian_policy(x, a, order, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = polynomial(x, order, act_dim)
    log_std = tf.get_variable(name=u'log_std', initializer=-1.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi


u"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope(u'pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope(u'v'):
        # DFK modified: want unbiased gradient estimate, so replacing MLP with
        # zero (for all states).
        # TODO(@eric): figure out how to do this right and not just multiply by zero.
        #v = tf.zeros_like(x)
        v = 0.0 * tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v

def polynomial_actor_critic(x, a, order, policy=None, action_space=None):
    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = polynomial_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        assert(False)

    with tf.variable_scope(u'pi',reuse=tf.AUTO_REUSE):
        pi, logp, logp_pi = policy(x, a, order, action_space)
    with tf.variable_scope(u'v',reuse=tf.AUTO_REUSE):
        # DFK modified: want unbiased gradient estimate, so replacing MLP with
        # zero (for all states).
        # TODO(@eric): figure out how to do this right and not just multiply by zero.
        v = 0.0 * tf.squeeze(mlp(x, list((1,1))+[1], tf.tanh, None), axis=1)
    return pi, logp, logp_pi, v
