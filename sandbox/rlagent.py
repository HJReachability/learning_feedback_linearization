import tensorflow as tf
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import copy
from scipy.linalg import solve_continuous_are

class RLAgent(object):
    #Initialize a reinforcement learning agent given its model
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
                 scaling,
                 state_constraint=None):
        self._num_iters = num_iters
        self._learning_rate = learning_rate
        self._desired_kl = desired_kl
        self._discount_factor = discount_factor
        self._num_rollouts = num_rollouts
        self._num_steps_per_rollout = num_steps_per_rollout
        self._num_total_time_steps = num_rollouts * num_steps_per_rollout
        self._dynamics = dynamics
        self._initial_state_sampler = initial_state_sampler
        self._feedback_linearization = feedback_linearization
        self._logger = logger
        self._state_constraint = state_constraint


         # Previous states and auxiliary controls. Used for KL update rule.
        self._previous_xs = None
        self._previous_vs = None
        self._previous_means = None
        self._previous_std = None
        self._norm = norm
        self._scaling =  scaling
        self.A,self.B, C=self._dynamics.linearized_system()

    def _update_feedback(self, rollouts):
        raise NotImplementedError()

    def run(self, plot=True, show_diff=False, dump_every=500):
        for ii in range(self._num_iters):
            print("---------- Iteration ", ii, " ------------")
            start_time = time.time()
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

            # Update stuff.
            self._update_feedback(rollouts)


            # Log elapsed time.
            elapsed_time = time.time() - start_time
            print("Elapsed time: ", elapsed_time)
            self._logger.log("elapsed_time", elapsed_time)

            # Dump every 'dump_every' iterations.
            if ii % dump_every == 1:
                self._logger.log("feedback_linearization", self._feedback_linearization)
                self._logger.dump()

        # Log the learned model.
        self._logger.log("feedback_linearization", self._feedback_linearization)

    def _collect_rollouts(self,time_step):
        rollouts = []
        num_total_time_steps = 0
        while num_total_time_steps < self._num_total_time_steps:
            # (0) Sample a new initial state.
            x = self._initial_state_sampler(time_step)

            # (1) Generate a time series for v and corresponding y.
            reference,K = self._generate_reference(x)
            ys, _ = self._generate_ys(x,reference,K)

            # (2) Push through dynamics and get x, y time series.
            rollout = {"xs" : [],
                       "us" : [],
                       "vs" : [],
                       "ys" : [],
                       "y_desireds" : [],
                       "rs" : []}

            for ref, y_desired in zip(reference, ys):

                next_y=self._dynamics.linearized_system_state(x)
                r = self._reward(y_desired, next_y)
                diff=self._dynamics.linear_system_state_delta(ref,next_y)
                v=-1*K @ (diff)
                u = self._feedback_linearization.sample_noisy_feedback(x, v)
                next_x = self._dynamics.integrate(x, u)

                if num_total_time_steps >= self._num_total_time_steps:
                    break

                rollout["xs"].append(x)
                rollout["vs"].append(v)
                rollout["us"].append(u)
                rollout["ys"].append(next_y)
                rollout["y_desireds"].append(y_desired)
                rollout["rs"].append(r)

                x = next_x
                num_total_time_steps += 1

                if self._state_constraint is not None and \
                   not self._state_constraint.contains(next_x):
                    rollout["rs"][-1] -= 100.0
                    break



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
    
    def _generate_reference(self, x0):
        """
        Use sinusoid with random frequency, amplitude, and bias:
              ``` vi(k) = a * sin(2 * pi * f * k) + b  ```
        """
        MAX_CONTINUOUS_TIME_FREQ = 0.1
        MAX_DISCRETE_TIME_FREQ = MAX_CONTINUOUS_TIME_FREQ * self._dynamics._time_step

        linsys_xdim=self.A.shape[0]
        linsys_udim=self.B.shape[1]
        Q=10.0 * (np.random.uniform() + 0.1) * np.eye(linsys_xdim)
        R=1.0 * (np.random.uniform() + 0.1) * np.eye(linsys_udim)

        # Initial y.
        y0 = self._dynamics.linearized_system_state(x0)

        y = np.empty((linsys_xdim, self._num_steps_per_rollout))
        for ii in range(linsys_xdim):
            y[ii, :] = np.linspace(
                0, self._num_steps_per_rollout * self._dynamics._time_step,
                self._num_steps_per_rollout)
            y[ii, :] = y0[ii, 0] + 1.0 * np.random.uniform() * (1.0 - np.cos(
                2.0 * np.pi * MAX_DISCRETE_TIME_FREQ * \
                np.random.uniform() * y[ii, :])) #+ 0.1 * np.random.normal()

        # Ensure that y ref starts at y0.
        assert(np.allclose(y[:, 0].flatten(), y0.flatten(), 1e-5))

        P = solve_continuous_are(self.A, self.B, Q, R)
        K = np.linalg.inv(R) @ self.B.T @ P
        return (np.split(y, indices_or_sections=self._num_steps_per_rollout, axis=1),K)

    def _generate_ys(self, x0, refs,K):
        """
        Compute desired output sequence given initial state and input sequence.
        This is computed by applying the true dynamics' feedback linearization.
        """
        x = x0.copy()
        ys = []
        xs = []
        for r in refs:
            y=self._dynamics.linearized_system_state(x)
            ys.append(y.copy())
            xs.append(x.copy())
            diff = self._dynamics.linear_system_state_delta(r, y)
            v = -K @ diff
            u = self._dynamics.feedback(x, v)
            x = self._dynamics.integrate(x, u)

        return ys, xs

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

    

   



    
    


