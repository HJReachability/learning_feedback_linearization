

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

    return torch.nn.Sequential(OrderedDict(layers))

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

             ``` u = [ M1(x) + M2(x) ] * [ (w1(x) + w2(x)) + v ] + noise ```

        where M1(x) and w1(x) are derived from modeled dynamics, and
        M2(x) and w2(x) are torch.Tensor objects that we will learn with policy
        gradients.
        """
        self._M1, self._w1 = dynamics.feedback_linearize()

        # Create some PyTorch tensors for unmodeled dynamics terms.
        self._M2 = create_network(
            dynamics.xdim, dynamics.udim**2,
            num_layers, num_hidden_units, activation)
        self._w2 = create_network(
            dynamics.xdim, dynamics.udim,
            num_layers, num_hidden_units, activation)

        self._xdim = dynamics.xdim
        self._udim = dynamics.udim

        # Set noise std.
        self._noise_std = noise_std

    def feedback(self, x, v):
        """ Compute u from x, v (np.arrays). See above comment for details. """
        v = np.reshape(v, (self._udim, 1))

        M = torch.from_numpy(self._M1(x)).float() + torch.reshape(
            self._M2(torch.from_numpy(x).float()), (self._udim, self._udim))
        w = torch.from_numpy(self._w1(x)).float() + torch.reshape(
            self._w2(torch.from_numpy(x).float()), (self._udim, 1))
        return torch.mm(M, w + torch.from_numpy(v).float())

    def noisy_feedback(self, x, v):
        """ Compute noisy version of u given x, v (np.arrays). """
        return torch.distributions.normal.Normal(
            self.feedback(x, v), self._noise_std)

    def log_prob(self, u, x, v):
        """ Compute log probability of u given x, v. """
        v = np.reshape(v, (self._udim, 1))
        u = np.reshape(u, (self._udim, 1))

        return torch.sum(self.noisy_feedback(x, v).log_prob(
            torch.from_numpy(u).float()))
