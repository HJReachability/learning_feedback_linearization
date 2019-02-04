import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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
        self._M2_optimizer = torch.optim.RMSprop(
            self._feedback_linearization._M2.parameters(),
            lr=self._learning_rate)
        self._w2_optimizer = torch.optim.RMSprop(
            self._feedback_linearization._w2.parameters(),
            lr=self._learning_rate)

    def run(self, plot=False):
        for ii in range(self._num_iters):
            print("---------- Iteration ", ii, " ------------")
            rollouts = self._collect_rollouts()

            if plot:
                # Plot a rollout's y vs y_desired.
                plt.clf()
                plt.figure(1)
                ys = rollouts[0]["ys"]
                y_desireds = rollouts[0]["y_desireds"]

                plt.plot(ys[0][0], ys[0][1], "b*")
                plt.plot([y[0] for y in ys], [y[1] for y in ys],
                         "b:", label="y")

                plt.plot(y_desireds[0][0], y_desireds[0][1], "r*")
                plt.plot([y[0] for y in y_desireds], [y[1] for y in y_desireds],
                         "r", label="y_desired")

                plt.xlabel("theta0")
                plt.ylabel("theta1")
                plt.legend()
                plt.pause(1)

            self._update_feedback(rollouts)

    def _collect_rollouts(self):
        rollouts = []
        for ii in range(self._num_rollouts):
#            print("===> New rollout")

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
                       "y_desireds" : [],
                       "rs" : []}

            for v, y_desired in zip(vs, ys):
                rollout["xs"].append(x)
                rollout["vs"].append(v)

                u = self._feedback_linearization.sample_noisy_feedback(x, v)
                rollout["us"].append(u)

                x = self._dynamics.integrate(x, u)
                y = self._dynamics.observation(x)
                rollout["ys"].append(y)
                rollout["y_desireds"].append(y_desired)

                r = self._reward(y_desired, y)
                rollout["rs"].append(r)

            # (3) Compute values for this rollout and append to list.
            self._compute_values(rollout)
            rollouts.append(rollout)

#        plt.figure()
#        xs = rollouts[0]["xs"]
#        theta1s = [x[0] for x in xs]
#        theta2s = [x[2] for x in xs]
#        plt.plot(theta1s, theta2s)
#        plt.pause(1)
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
        mean_return = 0.0
        for rollout in rollouts:
            for x, v, u, value in zip(rollout["xs"],
                                      rollout["vs"],
                                      rollout["us"],
                                      rollout["values"]):
                objective -= self._feedback_linearization.log_prob(
                    u, x, v) * value

            mean_return += rollout["values"][0]

        objective /= float(self._num_rollouts * self._num_steps_per_rollout)
        mean_return /= float(self._num_rollouts)

        print("Objective is: ", objective)
        print("Mean return is: ", mean_return)

        if torch.isnan(objective) or torch.isinf(objective):
            self._learning_rate /= 2.0

            for param_group in self._M2_optimizer.param_groups:
                param_group['lr'] = self._learning_rate
            for param_group in self._w2_optimizer.param_groups:
                param_group['lr'] = self._learning_rate

            print("=======> Oops. Objective was NaN or Inf. Please come again.")
            print("=======> Learning rate is now ", self._learning_rate)
            return

        # (2) Backpropagate derivatives.
        self._M2_optimizer.zero_grad()
        self._w2_optimizer.zero_grad()
        objective.backward()

        # (3) Update all learnable parameters.
        self._M2_optimizer.step()
        self._w2_optimizer.step()

    def _generate_v(self):
        """
        Use sinusoid with random frequency, amplitude, and bias:
              ``` vi(k) = a * sin(2 * pi * f * k) + b  ```
        """
        v = np.empty((self._dynamics.udim, self._num_steps_per_rollout))
        for ii in range(self._dynamics.udim):
            v[ii, :] = np.arange(self._num_steps_per_rollout)
            v[ii, :] = 5.0 * np.random.uniform(
                size=(1, self._num_steps_per_rollout)) * np.sin(
                2.0 * np.pi * 4.0 * np.random.uniform() * v[ii, :]) + np.random.normal()

        return np.split(
            v, indices_or_sections=self._num_steps_per_rollout, axis=1)

    def _generate_y(self, x0, v):
        """
        Compute desired output given initial state and input sequence:
                 ``` y(t) = h(x0) + \int \int v(t1) dt1 dt2 ```
        """
        initial_observation = self._dynamics.observation(x0)
#        print("init obs: ", initial_observation)

        v_array = np.concatenate(v, axis=1)

        # Append an extra 0 control at the end.
        v_array = np.concatenate([v_array, np.zeros(v[0].shape)], axis=1)
#        print("v", v_array)

        single_integrated_v = self._dynamics._time_step * np.cumsum(
            v_array, axis=1)
#        double_integrated_v = np.cumsum(
#            single_integrated_v, axis=1)
        double_integrated_v = self._dynamics._time_step * np.cumsum(
            single_integrated_v, axis=1)
#        print("single_v: ", single_integrated_v)
#        print("double_v: ", double_integrated_v)
#        print("double_v_norm: ", np.linalg.norm(double_integrated_v, axis=0))

        y = initial_observation + double_integrated_v[:, 1:]
        return np.split(
            y, indices_or_sections=self._num_steps_per_rollout, axis=1)

    def _reward(self, y_desired, y):
        return -np.linalg.norm(y_desired - y)**2

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
