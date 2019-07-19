from gym.envs.registration import register

register(
    id='Quadrotor14dEnv-v0',
    entry_point='quadrotor_14d_env.envs:Quadrotor14dEnv',
)