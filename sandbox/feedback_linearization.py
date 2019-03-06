

import torch
import numpy as np
from collections import OrderedDict

def shape2dims(shape):
    num_dims = 1
    for dim in shape:
        num_dims *= dim

    return num_dims

def create_network(num_inputs, num_outputs,
                   num_layers, num_hidden_units, activation):
    layers = [("L_initial_" + str(num_inputs) + "_" + str(num_hidden_units),
               torch.nn.Linear(num_inputs, num_hidden_units)),
              ("A_initial_" + str(num_inputs) + "_" + str(num_hidden_units),
               activation)]
    for ii in range(num_layers):
        layers.append((
            "L_" + str(ii) + "_" + str(num_hidden_units) +
            "_" + str(num_hidden_units),
            torch.nn.Linear(num_hidden_units, num_hidden_units)))
        layers.append((
            "A_" + str(ii) + "_" + str(num_inputs) +
            "_" + str(num_hidden_units),
            activation))

    layers.append((
        "L_final_" + str(num_hidden_units) + "_" + str(num_outputs),
            torch.nn.Linear(num_hidden_units, num_outputs)))
    layers.append((
        "A_final_" + str(num_outputs),
            torch.nn.Tanh()))

    return torch.nn.Sequential(OrderedDict(layers))

def init_weights(net, mean=0.0, std=0.1):
    for l in net:
        if type(l) in [torch.nn.Linear]:
            torch.nn.init.normal_(l.weight.data, mean, std)

class FeedbackLinearization(object):
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

        # Create some PyTorch tensors for unmodeled dynamics terms.
        self._M2 = create_network(
            dynamics.xdim, dynamics.udim**2,
            num_layers, num_hidden_units, activation)
        self._f2 = create_network(
            dynamics.xdim, dynamics.udim,
            num_layers, num_hidden_units, activation)

        # Initialize weights.
        init_weights(self._M2)
        init_weights(self._f2)

        self._xdim = dynamics.xdim
        self._udim = dynamics.udim

        # Set noise std.
        self._noise_std_variable = torch.ones((1, 1), requires_grad=True)
        self._noise_std = noise_std * self._noise_std_variable + 0.01

    def feedback(self, x, v):
        """ Compute u from x, v (np.arrays). See above comment for details. """
        v = np.reshape(v, (self._udim, 1))

        # Scaling factor to scale output of tanh layers.
        SCALING = 4.0

        M = torch.from_numpy(self._M1(x)).float() + torch.reshape(
            SCALING * self._M2(torch.from_numpy(x.flatten()).float()),
            (self._udim, self._udim))
        f = torch.from_numpy(self._f1(x)).float() + torch.reshape(
            SCALING * self._f2(torch.from_numpy(x.flatten()).float()),
            (self._udim, 1))

        # TODO! Make sure this is right (and consistent with dynamics.feedback).
        return torch.mm(M, torch.from_numpy(v).float()) + f

    def noisy_feedback(self, x, v):
        """ Compute noisy version of u given x, v (np.arrays). """
        return torch.distributions.normal.Normal(
            self.feedback(x, v), self._noise_std)

    def sample_noisy_feedback(self, x, v):
        """ Compute noisy version of u given x, v (np.arrays). """
        return self.noisy_feedback(x, v).sample().numpy()

    def log_prob(self, u, x, v):
        """ Compute log probability of u given x, v. """
        v = np.reshape(v, (self._udim, 1))
        u = np.reshape(u, (self._udim, 1))

        return torch.sum(self.noisy_feedback(x, v).log_prob(
            torch.from_numpy(u).float()))

    def prob(self, u, x, v):
        """ Compute probability of u given x, v. """
        v = np.reshape(v, (self._udim, 1))
        u = np.reshape(u, (self._udim, 1))

        return torch.sum(self.noisy_feedback(x, v).prob(
            torch.from_numpy(u).float()))
