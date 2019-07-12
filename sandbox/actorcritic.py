from rlagent import RLAgent
import tensorflow as tf
from tf_feedback_linearization import create_network
import numpy as np

class ActorCritic(RLAgent):
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
        super(ActorCritic,self).__init__(num_iters,
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
        self._critic = create_network(feedback_linearization._num_inputs,1,
            feedback_linearization._num_layers, feedback_linearization._num_hidden_units,
            feedback_linearization._activation, "linear")

        self._critic_loss = tf.keras.losses.MeanSquaredError()
    
        self._M2_optimizer = tf.keras.optimizers.RMSprop(self._learning_rate)

        self._f2_optimizer = tf.keras.optimizers.RMSprop(self._learning_rate)

        self._critic_optimizer = tf.keras.optimizers.RMSprop(10*self._learning_rate)
    def _update_feedback(self, rollouts):
        """
        Uthe feedback law contained in self._feedback_linearization
        based on the observed rollouts.
        """

        #(1) Evaluate and train Critic
        mean_critic_loss = 0.0
        for rollout in rollouts:
            mean_critic_loss += self._train_value_function(rollout)
        mean_critic_loss /= float(self._num_rollouts)

        # (2) Construct the objective.
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

                mean_return += rollout["values"][0]

            objective /= float(self._num_total_time_steps)
            mean_return /= float(self._num_rollouts)

        print("Objective is: ", objective)
        print("Mean return is: ", mean_return)
        print("Mean critic loss is: ", mean_critic_loss)

        

        stddev = 0.1 + self._feedback_linearization._noise_scaling 
        print("Std is: ", stddev)

        self._logger.log("mean_return", mean_return)
        self._logger.log("mean_critic_loss", mean_critic_loss)
        self._logger.log("learning_rate", self._learning_rate)
        self._logger.log("stddev", stddev)
        self._logger.log("noise_std_variable", self._feedback_linearization._noise_std_variable.numpy())
    
      
        m2gradient = tape.gradient(objective,self._feedback_linearization._M2_net.trainable_variables)
        f2gradient = tape.gradient(objective,self._feedback_linearization._f2_net.trainable_variables)
        noisegradient = tape.gradient(objective,self._feedback_linearization._noise_std_variable)


        self._M2_optimizer.apply_gradients(zip(m2gradient,self._feedback_linearization._M2_net.trainable_variables))
        self._f2_optimizer.apply_gradients(zip(f2gradient,self._feedback_linearization._f2_net.trainable_variables))
        self._feedback_linearization._noise_std_variable.assign_sub(10*self._learning_rate*noisegradient)

        del tape

    
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

    def _compute_advantages(self, rollout):
        """ Baseline raw values. """
        values = rollout["values"]
        states = rollout["xs"]
        baselined = []

        predictedvalues = []
        for i in range(len(states)):
            predictedvalues.append(self._critic(self._dynamics.preprocess_state(states[i])))
            baselined.append(values[i]-predictedvalues[i].numpy()[0][0])
        rollout["predictedvalues"] = predictedvalues
        if(self._iter_count<100):
            avg = sum(values) / float(len(values))
            baselined = [(v - avg)  for v in values]
        rollout["advantages"] = baselined
        
    
    def _train_value_function(self, rollout):
        with tf.GradientTape() as tape:
            self._compute_advantages(rollout)
            loss = self._critic_loss(rollout["values"], rollout["predictedvalues"])
        criticgradient = tape.gradient(loss, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(zip(criticgradient,self._critic.trainable_variables))
        return loss