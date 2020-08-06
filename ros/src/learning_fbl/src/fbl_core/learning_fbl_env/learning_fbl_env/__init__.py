from gym.envs.registration import register

register(
    id='LearningFBLEnv-v0',
    entry_point='learning_fbl_env.envs:LearningFBLEnv'
#    kwargs={'stepsPerRollout' : 25, 'rewardScaling' : 10, 'dynamicsScaling' : 0.33, 'preprocessState' : 1,
#    'uscaling' : 0.1, 'largerQ' : 1},
)
