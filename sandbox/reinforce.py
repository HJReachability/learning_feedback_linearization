import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import copy
from feedback_linearization import FeedbackLinearization
from dynamics import Dynamics

class Reinforce(object):
    def __init__(self,
                 num_iters,
                 learning_rate,
                 desired_kl,
                 discount_factor,
                 num_rollouts,
                 num_steps_per_rollout,
                 dynamics,
                 initial_state_sampler,
                 feedback_linearization,
                 logger,
                 norm,
                 scaling):
        self._num_iters = num_iters
        self._learning_rate = learning_rate
        self._desired_kl = desired_kl
        self._discount_factor = discount_factor
        self._num_rollouts = num_rollouts
        self._num_steps_per_rollout = num_steps_per_rollout
        self._dynamics = dynamics
        self._initial_state_sampler = initial_state_sampler
        self._feedback_linearization = feedback_linearization
        self._logger = logger

        # Use RMSProp as the optimizer.
        self._M2_optimizer = torch.optim.Adam(
            self._feedback_linearization._M2_net.parameters(),
            lr=self._learning_rate)
#            momentum=0.8,
#            weight_decay=0.0001)
        self._f2_optimizer = torch.optim.Adam(
            self._feedback_linearization._f2_net.parameters(),
            lr=self._learning_rate)
#            momentum=0.8,
#            weight_decay=0.0001)
        self._noise_std_optimizer = torch.optim.Adam(
            [self._feedback_linearization._noise_std_variable],
            lr=self._learning_rate*10)

        # Previous states and auxiliary controls. Used for KL update rule.
        self._previous_xs = None
        self._previous_vs = None
        self._previous_means = None
        self._previous_std = None
        self._norm = norm
        self._scaling =  scaling

    def run(self, plot=False, show_diff=False):
        for ii in range(self._num_iters):
            print("---------- Iteration ", ii, " ------------")
            rollouts = self._collect_rollouts(ii)

            ys = rollouts[0]["ys"]
            y_desireds = rollouts[0]["y_desireds"]

            if plot:
                plt.clf()
                plt.figure(1)
                plt.plot(ys[0][0], ys[0][1], "b*")
                plt.plot([y[0] for y in ys], [y[1] for y in ys],
                         "b:", label="y")

                plt.plot(y_desireds[0][0], y_desireds[0][1], "r*")
                plt.plot([y[0] for y in y_desireds], [y[1] for y in y_desireds],
                         "r", label="y_desired")

                plt.xlabel("theta0")
                plt.ylabel("theta1")
                plt.legend()
                plt.pause(0.01)

            # Log these trajectories.
            self._logger.log("ys", ys)
            self._logger.log("y_desireds", y_desireds)

            # Every 10 steps, log a deepcopy of the full feedback linearization.
#            if ii % 10:
#                self._logged_feedback_linearization = copy.deepcopy(
#                    self._feedback_linearization)
#                self._logger.log(
#                    "feedback_linearization", self._logged_feedback_linearization)

            # Update stuff.
            self._update_feedback(rollouts)

            if show_diff:
                newM=[weight.detach().numpy() for weight in list(self._feedback_linearization._M2_net.parameters())]
                newF=[weight.detach().numpy() for weight in list(self._feedback_linearization._f2_net.parameters())]

                # Prematurely dump in case we terminate early.

                if ii>0:
                    diffM=np.sum([np.linalg.norm(old-new)**2 for old,new in zip(oldweightsM,newM)])
                    difff=np.sum([np.linalg.norm(old-new)**2 for old,new in zip(oldweightsf,newF)])

                    print (diffM,difff)

                oldweightsM=copy.deepcopy(newM)
                oldweightsf=copy.deepcopy(newF)
            self._logger.dump()

        # Log the learned model.
        self._logger.log("feedback_linearization", self._feedback_linearization)

    def _collect_rollouts(self,time_step):
        rollouts = []
        for ii in range(self._num_rollouts):
            # (0) Sample a new initial state.
            x = self._initial_state_sampler(time_step)

            # (1) Generate a time series for v and corresponding y.
            vs = self._generate_vs()
            ys = self._generate_ys(x, vs)

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
            self._compute_advantages(rollout)
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
            for x, v, u, adv in zip(rollout["xs"],
                                    rollout["vs"],
                                    rollout["us"],
                                    rollout["advantages"]):
                objective -= self._feedback_linearization.log_prob(
                    u, x, v) * adv

            mean_return += rollout["values"][0]

        objective /= float(self._num_rollouts * self._num_steps_per_rollout)
        mean_return /= float(self._num_rollouts)

        print("Objective is: ", objective)
        print("Mean return is: ", mean_return)
        print("Std is: ", self._feedback_linearization._noise_std_variable.data[0].detach().numpy()[0])

        self._logger.log("mean_return", mean_return)
        self._logger.log("learning_rate", self._learning_rate)
        self._logger.log(
            "stddev", self._feedback_linearization._noise_std_variable.data[0].detach().numpy()[0])

        if torch.isnan(objective) or torch.isinf(objective):
            for param_group in self._M2_optimizer.param_groups:
                param_group['lr'] /= 1.5
            for param_group in self._f2_optimizer.param_groups:
                param_group['lr'] /= 1.5
            for param_group in self._feedback_linearization._noise_std_optimizer.param_groups:
                param_group['lr'] /= 1.5

            print("=======> Oops. Objective was NaN or Inf. Please come again.")
            print("=======> Learning rate is now ", self._learning_rate)
            return

        # (2) Backpropagate derivatives.
        self._M2_optimizer.zero_grad()
        self._f2_optimizer.zero_grad()
        self._noise_std_optimizer.zero_grad()
        objective.backward(retain_graph=True)

        # (3) Update all learnable parameters.
        self._M2_optimizer.step()
        self._f2_optimizer.step()
        self._noise_std_optimizer.step()


        # (4) Update learning rate according to KL divergence criterion.
        if self._desired_kl > 0.0 and self._previous_xs is not None:
            kl = self._compute_kl()
            lr_scaling = None
            if kl > 2.0 * self._desired_kl:
                lr_scaling = 1.0 / 1.5
                self._learning_rate *= lr_scaling
                print("DECREASING learning rate to ", self._learning_rate)
            elif kl < 0.5 * self._desired_kl:
                lr_scaling = 1.5
                self._learning_rate *= lr_scaling
                print("INCREASING learning rate to ", self._learning_rate)

            if lr_scaling is not None:
                for param_group in self._M2_optimizer.param_groups:
                    param_group['lr'] *= lr_scaling
                for param_group in self._f2_optimizer.param_groups:
                    param_group['lr'] *= lr_scaling
                for param_group in self._feedback_linearization._noise_std_optimizer.param_groups:
                    param_group['lr'] /= 1.5


        # Update previous states visited and auxiliary control inputs.
        if self._previous_xs is None:
            self._previous_xs = np.zeros((
                self._num_rollouts * self._num_steps_per_rollout,
                self._dynamics.xdim))
            self._previous_vs = np.zeros((
                self._num_rollouts * self._num_steps_per_rollout,
                self._dynamics.udim))
            self._previous_means = np.zeros((
                self._num_rollouts * self._num_steps_per_rollout,
                self._dynamics.udim))

        ii = 0
        self._previous_std = self._feedback_linearization._noise_std_variable.data[0].detach().numpy()[0]
        for r in rollouts:
            for x, v in zip(r["xs"], r["vs"]):
                self._previous_xs[ii, :] = x.flatten()
                self._previous_vs[ii, :] = v.flatten()
                u = self._feedback_linearization.feedback(x, v).detach().numpy()
                self._previous_means[ii, :] = u.flatten()
                ii += 1

    def _generate_vs(self):
        """
        Use sinusoid with random frequency, amplitude, and bias:
              ``` vi(k) = a * sin(2 * pi * f * k) + b  ```
        """
        MAX_CONTINUOUS_TIME_FREQ = 2.0
        MAX_DISCRETE_TIME_FREQ = MAX_CONTINUOUS_TIME_FREQ * self._dynamics._time_step

        v = np.empty((self._dynamics.udim, self._num_steps_per_rollout))
        for ii in range(self._dynamics.udim):
            v[ii, :] = np.arange(self._num_steps_per_rollout)
            v[ii, :] = 1.0 * np.random.uniform(
                size=(1, self._num_steps_per_rollout)) * np.sin(
                2.0 * np.pi * MAX_DISCRETE_TIME_FREQ * np.random.uniform() * v[ii, :]) + \
                0.1 * np.random.normal()

        return np.split(
            v, indices_or_sections=self._num_steps_per_rollout, axis=1)

    def _generate_ys(self, x0, vs):
        """
        Compute desired output sequence given initial state and input sequence.
        This is computed by applying the true dynamics' feedback linearization.
        """
        x = x0.copy()
        ys = []
        for v in vs:
            u = self._dynamics.feedback(x, v)
            x = self._dynamics.integrate(x, u)
            ys.append(self._dynamics.observation(x))

        return ys

    def _reward(self, y_desired, y):
        return -self._scaling * self._dynamics.observation_distance(
            y_desired, y, self._norm)

    def _compute_values(self, rollout):
        """ Add a sum of future discounted rewards field to rollout dict."""
        rs = rollout["rs"]
        values = deque()
        last_value = 0.0
        for ii in range(len(rs)):
            # ii is a reverse iteration index, i.e. we're counting backward.
            r = rs[len(rs) - ii - 1]
            values.appendleft(self._discount_factor * last_value + r)
            last_value = values[0]

        # Convert to list.
        values = list(values)
        rollout["values"] = values

    def _compute_advantages(self, rollout):
        """ Baseline raw values. """
        values = rollout["values"]
        avg = sum(values) / float(len(values))
        std = np.std(np.array(values))
        baselined = [(v - avg)  for v in values]
        rollout["advantages"] = baselined

    def _compute_kl(self):
        """
        Compute average KL divergence between action distributions of current
        and previous policies (averaged over states in the current experience
        batch).

        NOTE: this is KL(old || new). We've implemented it just for Gaussians,
        the closed form for which may be found at:
        https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions
        """
        if self._previous_means is None:
            return 0.0

        num_states = self._num_rollouts * self._num_steps_per_rollout
        num_dimensions = self._dynamics.udim
        current_means = np.zeros((num_states, num_dimensions))
        for ii in range(num_states):
            x = np.reshape(self._previous_xs[ii, :], (self._dynamics.xdim, 1))
            v = np.reshape(self._previous_vs[ii, :], (self._dynamics.udim, 1))
            u = self._feedback_linearization.feedback(x, v).detach().numpy()
            current_means[ii, :] = u.copy().flatten()

        current_std = self._feedback_linearization._noise_std.item()
        current_cov = current_std**2 * np.ones(self._dynamics.udim)
        previous_cov = self._previous_std**2 * np.ones(self._dynamics.udim)

        means_diff = current_means - self._previous_means
        kl = 0.5 * (np.sum(previous_cov / current_cov) - num_dimensions +
                    np.sum(np.log(current_cov)) - np.sum(np.log(previous_cov)) +
                    np.mean(np.sum((means_diff / current_cov) * means_diff,
                                   axis = 1)))
        return kl
