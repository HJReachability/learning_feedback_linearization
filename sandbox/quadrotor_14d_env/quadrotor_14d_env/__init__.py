from gym.envs.registration import register

register(
    id='Quadrotor14dEnv-v0',
    entry_point='quadrotor_14d_env.envs:Quadrotor14dEnv',
    kwargs={'stepsPerRollout' : 25, 'rewardScaling' : 10, 'dynamicsScaling' : 0.33, 'preprocessState' : 1,
    'uscaling' : 0.1, 'largerQ' : 1},
)