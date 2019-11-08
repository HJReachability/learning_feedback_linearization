from gym.envs.registration import register

register(
    id='Quadrotor14dEnv-v0',
    entry_point='custom_envs.envs:Quadrotor14dEnv',
    kwargs={
        'stepsPerRollout' : 25,
        'rewardScaling' : 10,
        'dynamicsScaling' : 0.33,
        'preprocessState' : 1,
        'uscaling' : 0.1,
        'largerQ' : 1
    },
)

register(
    id='DoublePendulumEnv-v0',
    entry_point='custom_envs.envs:DoublePendulumEnv',
    kwargs={
        'stepsPerRollout' : 25,
        'rewardScaling' : 10,
        'dynamicsScaling' : 0.33,
        'preprocessState' : 1,
        'uscaling' : 0.1,
        'largerQ' : 1
    },
)

register(
    id='DiffDriveEnv-v0',
    entry_point='custom_envs.envs:DiffDriveEnv',
    kwargs={
        'stepsPerRollout' : 25,
        'rewardScaling' : 10,
        'dynamicsScaling' : 0.33,
        'preprocessState' : 1,
        'uscaling' : 1,
        'largerQ' : 1
    },
)

register(
    id='BallAndBeamEnv-v0',
    entry_point='custom_envs.envs:BallAndBeamEnv',
    kwargs={
        'stepsPerRollout' : 25,
        'rewardScaling' : 10,
        'uscaling' : 1,
    },
)
