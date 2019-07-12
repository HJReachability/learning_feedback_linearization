import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import OrderedDict



def create_network(num_inputs, num_outputs,
                   num_layers, num_hidden_units, activation, lastactivation = "tanh"):
    layers = []
    for i in range(num_layers):
        layers.append(tf.keras.layers.Dense(num_hidden_units, activation))

    layers.append(tf.keras.layers.Dense(num_outputs, lastactivation))
    print(lastactivation)

    model = tf.keras.models.Sequential(layers)
    return model

class TFFeedbackLinearization(object):
    def __init__(self,
                 dynamics,
                 num_layers,
                 num_hidden_units,
                 activation,
                 noise_std):
        """
        Initialize from a Dynamics object.
        This will encode the following control law:

             ``` u = [ M1(x) + M2(x) ] * v + [ f1(x) + f2(x) ] + noise ```

        where M1(x) and f1(x) are derived from modeled dynamics, and
        M2(x) and f2(x) are torch.Tensor objects that we will learn with policy
        gradients.
        """
        self._M1, self._f1 = dynamics.feedback_linearize()

        self._num_inputs = dynamics.preprocessed_xdim
        self._num_layers = num_layers
        self._num_hidden_units = num_hidden_units
        self._activation = activation


        # Create some Tensorflow NN for unmodeled dynamics terms.
        self._M2_net = create_network(
            dynamics.preprocessed_xdim, dynamics.udim**2,
            num_layers, num_hidden_units, activation)
        self._f2_net = create_network(
            dynamics.preprocessed_xdim, dynamics.udim,
            num_layers, num_hidden_units, activation)

        self._M2 = lambda x: self._M2_net(dynamics.preprocess_state(x))
        self._f2 = lambda x: self._f2_net(dynamics.preprocess_state(x))

        self._xdim = dynamics.xdim
        self._udim = dynamics.udim

        

        # Set noise std.
        self._noise_std_variable = tf.Variable(tf.ones((1, 1),dtype = tf.dtypes.float32))
        self._noise_scaling = noise_std

        

    def feedback(self, x, v):
        """ Compute u from x, v (np.arrays). See above comment for details. """
        v = np.reshape(v, (self._udim, 1))

        M = tf.cast(tf.convert_to_tensor(self._M1(x)),dtype = tf.float32) + tf.cast(tf.reshape(self._M2(x.flatten()),
            (self._udim, self._udim)), dtype = tf.float32)

        f = tf.cast(tf.convert_to_tensor(self._f1(x)),dtype = tf.float32) + tf.cast(tf.reshape(
            self._f2(x.flatten()),
            (self._udim, 1)),dtype = tf.float32)

        # TODO! Make sure this is right (and consistent with dynamics.feedback).
        return tf.cast(tf.matmul(M,tf.cast((tf.convert_to_tensor(v)),dtype = tf.float32)),dtype = tf.float32) + f
        
    def noisy_feedback(self, x, v):
        """ Compute noisy version of u given x, v (np.arrays). """
        mu = self.feedback(x,v)
        sigma = self._noise_scaling * tf.math.abs(self._noise_std_variable)
        distribution = tfp.distributions.Normal(mu,sigma)
        return distribution

    def sample_noisy_feedback(self, x, v):
        """ Compute noisy version of u given x, v (np.arrays). """
        return self.noisy_feedback(x, v).sample()

    def log_prob(self, u, x, v):
        """ Compute log probability of u given x, v. """
        v = np.reshape(v, (self._udim, 1))
        u = np.reshape(u, (self._udim, 1))


        return self.noisy_feedback(x, v).log_prob(tf.cast(tf.convert_to_tensor(u),dtype = tf.float32))


