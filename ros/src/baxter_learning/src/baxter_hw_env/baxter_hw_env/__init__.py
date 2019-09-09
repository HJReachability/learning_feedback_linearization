from gym.envs.registration import register

register(
    id='BaxterHwEnv-v0',
    entry_point='baxter_hw_env.envs:BaxterHwEnv'
#    kwargs={'stepsPerRollout' : 25, 'rewardScaling' : 10, 'dynamicsScaling' : 0.33, 'preprocessState' : 1,
#    'uscaling' : 0.1, 'largerQ' : 1},
)
