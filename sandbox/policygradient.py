from rlagent import RLAgent
import tensorflow as tf
import numpy as np

class PolicyGradient(RLAgent):
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
        super(PolicyGradient,self).__init__(num_iters,
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
                 state_constraint=None)
    def _update_feedback(self, rollouts):
        """
        Update the feedback law contained in self._feedback_linearization
        based on the observed rollouts.
        """
        # (1) Construct the objective.
        #NOTE: this could probably be done more efficiently using some fancy
        # tensor arithmetic.
        objective = tf.zeros(1)
        mean_return = 0.0
        with tf.GradientTape(persistent = True) as tape:
            for rollout in rollouts:
                for x, v, u, adv in zip(rollout["xs"],
                                        rollout["vs"],
                                        rollout["us"],
                                        rollout["advantages"]):
                    objective -= self._feedback_linearization.log_prob(
                        u, x, v) * adv

                mean_return -= rollout["values"][0]

            objective /= float(self._num_total_time_steps)
            mean_return /= float(self._num_rollouts)

        print("Objective is: ", objective)
        print("Mean return is: ", mean_return)
        

        stddev = 0.1 + self._feedback_linearization._noise_scaling 
        print("Std is: ", stddev)

        self._logger.log("mean_return", mean_return)
        self._logger.log("learning_rate", self._learning_rate)
        self._logger.log("stddev", stddev)

        
        m2gradient = tape.gradient(objective,self._feedback_linearization._M2_net.trainable_variables)
        f2gradient = tape.gradient(objective,self._feedback_linearization._f2_net.trainable_variables)
        noisegradient = tape.gradient(objective,self._feedback_linearization._noise_std_variable)

        for i in range(len(m2gradient)):
            self._feedback_linearization._M2_net.trainable_variables[i].assign_add(self._learning_rate*m2gradient[i])
        for i in range(len(f2gradient)):
            self._feedback_linearization._f2_net.trainable_variables[i].assign_add(self._learning_rate*f2gradient[i])
        self._feedback_linearization._noise_std_variable.assign_add(self._learning_rate*noisegradient)

        del tape

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
        

        if self._previous_xs is None:
            self._previous_xs = np.zeros((
                self._num_total_time_steps, self._dynamics.xdim))
            self._previous_vs = np.zeros((
                self._num_total_time_steps, self._dynamics.udim))
            self._previous_means = np.zeros((
                self._num_total_time_steps, self._dynamics.udim))

        ii = 0
        self._previous_std = self._feedback_linearization._noise_std_variable.numpy()[0]
        for r in rollouts:
            for x, v in zip(r["xs"], r["vs"]):
                self._previous_xs[ii, :] = x.flatten()
                self._previous_vs[ii, :] = v.flatten()
                u = self._feedback_linearization.feedback(x, v).numpy()
                self._previous_means[ii, :] = u.flatten()
                ii += 1

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

        num_states = self._num_total_time_steps
        num_dimensions = self._dynamics.udim
        current_means = np.zeros((num_states, num_dimensions))
        for ii in range(num_states):
            x = np.reshape(self._previous_xs[ii, :], (self._dynamics.xdim, 1))
            v = np.reshape(self._previous_vs[ii, :], (self._dynamics.udim, 1))
            u = self._feedback_linearization.feedback(x, v).numpy()
            current_means[ii, :] = u.copy().flatten()

        current_std = self._feedback_linearization._noise_scaling * \
                      abs(self._feedback_linearization._noise_std_variable.item()) + 0.1
        current_cov = current_std**2 * np.ones(self._dynamics.udim)
        previous_cov = self._previous_std**2 * np.ones(self._dynamics.udim)

        means_diff = current_means - self._previous_means
        kl = 0.5 * (np.sum(previous_cov / current_cov) - num_dimensions +
                    np.sum(np.log(current_cov)) - np.sum(np.log(previous_cov)) +
                    np.mean(np.sum((means_diff / current_cov) * means_diff,
                                   axis = 1)))
        return kl


    
