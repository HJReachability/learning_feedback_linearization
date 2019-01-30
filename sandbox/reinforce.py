


import torch
import numpy as np
from collections import deque

from feedback_linearization import FeedbackLinearization
from dynamics import Dynamics

class Reinforce(object):
    def __init__(self,
                 num_iters,
                 learning_rate,
                 discount_factor,
                 num_rollouts,
                 num_steps_per_rollout,
                 dynamics,
                 initial_state_sampler,
                 feedback_linearization):
        self._num_iters = num_iters
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._num_rollouts = num_rollouts
        self._num_steps_per_rollout = num_steps_per_rollout
        self._dynamics = dynamics
        self._initial_state_sampler = initial_state_sampler
        self._feedback_linearization = feedback_linearization

        # Use RMSProp as the optimizer.
        self._optimizer = torch.optim.RMSprop([
            self._feedback_linearization._M2.parameters(),
            self._feedback_linearization._w2.parameters()],
                                              lr=self._learning_rate)

    def run(self):
        for ii in range(self._num_iters):
            rollouts = self._collect_rollouts()
            self._update_feedback(rollouts)

    def _collect_rollouts(self):
        rollouts = []
        for ii in range(self._num_rollouts):
            # (0) Sample a new initial state.
            x = self._initial_state_sampler()

            # (1) Generate a time series for v and corresponding y.
            vs = self._generate_v()
            ys = self._generate_y(x, vs)

            # (2) Push through dynamics and get x, y time series.
            rollout = {"xs" : [],
                       "us" : [],
                       "vs" : [],
                       "ys" : [],
                       "rs" : []}

            for v, y_desired in zip(vs, ys):
                rollout["xs"].append(x)
                rollout["vs"].append(v)

                u = self._feedback_linearization.noisy_feedback(x, v)
                rollout["us"].append(u)

                x = self._dynamics.integrate(x, u)
                y = self._dynamics.observation(x)
                rollout["ys"].append(y)

                r = self._reward(y_desired, y)
                rollout["rs"].append(r)

            # (3) Compute values for this rollout and append to list.
            self._compute_values(rollout)
            rollouts.append(rollout)

        return rollouts

    def _update_feedback(self, rollouts):
        """
        Update the feedback law contained in self._feedback_linearization
        based on the observed rollouts.
        """
        # (1) Construct the objective.
        # NOTE: this could probably be done more efficiently using some fancy
        # tensor arithmetic.
        objective = torch.zeros(1)
        for rollout in rollouts:
            for x, v, u, value in zip(rollout["xs"],
                                      rollout["vs"],
                                      rollout["us"],
                                      rollout["values"]):
                objective += self._feedback_linearization.log_prob(
                    u, x, v) * value

        objective /= float(self._num_rollouts * self._num_steps_per_rollout)

        print("Objective is: ", objective)

        # (2) Backpropagate derivatives.
        self._optimizer.zero_grad()
        objective.backward()

        # (3) Update all learnable parameters.
        self._optimizer.step()

    def _generate_v(self):
        pass

    def _generate_y(self, x0, v):
        pass

    def _reward(self, y_desired, y):
        return np.linalg.norm(y_desired - y)**2

    def _compute_values(self, rollout):
        """ Add a sum of future discounted rewards field to rollout dict."""
        rs = rollout["rs"]
        values = deque()
        last_value = 0.0
        for ii in range(len(rs)):
            # ii is a reverse iteration index, i.e. we're counting backward.
            r = rs[len(rs) - ii - 1]
            values.appendleft(self._discount_factor * last_value + r)

        rollout["values"] = list(values)
