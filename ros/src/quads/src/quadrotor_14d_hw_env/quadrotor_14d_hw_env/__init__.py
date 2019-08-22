from gym.envs.registration import register

register(
    id='Quadrotor14dHwEnv-v0',
<<<<<<< HEAD
    entry_point='quadrotor_14d_env.envs:Quadrotor14dHwEnv',
=======
    entry_point='quadrotor_14d_hw_env.envs:Quadrotor14dHwEnv'
>>>>>>> 2f3031e323d5fea502f4d2b06156fa58d5edcffd
#    kwargs={'stepsPerRollout' : 25, 'rewardScaling' : 10, 'dynamicsScaling' : 0.33, 'preprocessState' : 1,
#    'uscaling' : 0.1, 'largerQ' : 1},
)
