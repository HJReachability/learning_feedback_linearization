
import fbl_core
from geo_quad_fbl_class import TwoControllerQuadFBL
from geo_quad_dynamics import GeoQuadDynamics 
from geo_quad_reference_generator import ConstantTwistQuadTrajectory
from pos_att_controller import PosAttController 
from fbl_core.controller import PD 

import numpy as np

import os
import inspect
import pickle


PREFIX = os.path.abspath(os.path.join(os.path.dirname(inspect.getfile(fbl_core)), os.pardir, os.pardir, 'data'))



def fill_parameters():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['baseline', 'train'], default='baseline')
    parser.add_argument('--exp_name', type=str, default='todo')
    args = parser.parse_args()
    params = vars(args)

    # Description
    params['env_type'] = 'TwoControllerQuadFBL'
    params['controller_type'] = 'PosAttController'

    # Dynamics
    params['nominal_m'] = 4.34
    params['nominal_J'] = [0.0820, 0.0845, 0.1377]
    params['nominal_ls'] = [0.315, 0.315, 0.315, 0.315]
    params['nominal_ctfs'] = [8.004e-4, 8.004e-4, 8.004e-4, 8.004e-4]

    params['true_m'] = 4.0
    params['true_J'] = [0.067, 0.063, 1.24]
    params['true_ls'] = [0.3145, 0.316, 0.315, 0.313]
    params['true_ctfs'] = [8e-4, 8.2e-4, 7.91e-4, 8.002e-4]

    # Controller Params
    params['controller_description'] = 'two PD controllers'
    params['pos_p_gain'] = 16*4.34
    params['pos_v_gain'] = 5.6*4.34
    params['att_p_gain'] = 8.81
    params['att_v_gain'] = 2.54

    # Reference
    params['ref_type'] = 'constant_twist'
    params['nominal_traj_length'] = 20

    # FBL class
    params['action_scaling'] = 1
    params['reward_scaling'] = 1
    params['reward_norm'] = 1

    # State Limits
    params['init_limits'] = np.array([8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5])
    params['obs_limits'] = np.array([10,10,10,1,1,1,1,1,1,1,1,1,5,5,5,5,5,5])
    params['action_limits'] = 50

    # Learning
    params['learning_alg'] = 'ppo'
    params['learning_rate'] = 3e-4
    params['n_steps'] = 2048
    params['batch_size'] = 64
    params['n_epochs'] = 10
    params['gamma'] = 0.99
    params['gae_lambda'] = 0.95
    params['total_timesteps'] = 3e5


    return params


def create_environment(params):

    import gym
    from gym import spaces
    import learning_fbl_env
    
    # Create Dynamics
    nominal_dynamics = GeoQuadDynamics(params['nominal_m'], params['nominal_J'], params['nominal_ls'], params['nominal_ctfs'])
    true_dynamics = GeoQuadDynamics(params['true_m'], params['true_J'], params['true_ls'], params['true_ctfs'])

    # Create Controller
    pos_controller = PD(3,3,params['pos_p_gain'], params['pos_v_gain'])
    att_controller = PD(3,3,params['att_p_gain'], params['att_v_gain'])
    controller = PosAttController(pos_controller, att_controller)

    # Create Reference
    if params['ref_type'] == 'constant_twist':
        reference_generator = ConstantTwistQuadTrajectory(params['init_limits'], -params['init_limits'], 
                                                          nominal_dynamics, params['nominal_traj_length'])

    # Create action and observation spaces
    udim = nominal_dynamics.udim
    xdim = nominal_dynamics.xdim

    action_space = spaces.Box(low=-params['action_limits'], high=params['action_limits'], shape=(udim**2 + udim,), dtype=np.float32)
    observation_space = spaces.Box(low=-params['obs_limits'], high=params['obs_limits'], shape=(xdim,), dtype=np.float32)


    # Create FBL Class
    fbl_obj = TwoControllerQuadFBL(action_space = action_space, observation_space = observation_space,
        nominal_dynamics = nominal_dynamics, true_dynamics = true_dynamics,
        cascaded_quad_controller = controller,
        reference_generator = reference_generator,
        action_scaling = params['action_scaling'], reward_scaling = params['reward_scaling'], reward_norm = params['reward_norm'])


    # Create environment
    # env = lambda : gym.make('learning_fbl_env:LearningFBLEnv-v0', learn_fbl = fbl_obj)
    env = gym.make('learning_fbl_env:LearningFBLEnv-v0', learn_fbl = fbl_obj)

    return env


def train(env, params, logdir):

    if params['learning_alg'] == ppo:
        from stable_baselines3 import PPO
        from stable_baselines3.ppo import MlpPolicy

        model = PPO(MlpPolicy, env, verbose = 1,
                learning_rate = params['learning_rate'],
                n_steps = params['n_steps'],
                batch_size = params['batch_size'],
                n_epochs = params['n_epochs'],
                gamma = params['gamma'],
                gae_lambda = params['gae_lambda'],
                tensorboard_log = logdir
            )
    
    with open(logdir + 'params.pkl', 'wb') as f:
        pickle.dump(params, f, 0)

    model.learn(total_timesteps = params['total_timesteps'])
    model.save(logdir)


    



if __name__ == '__main__':

    params = fill_parameters()
    env = create_environment(params)

    exp_dir_name = 'geo_quad_' + params['exp_name']
    expdir = os.path.abspath(os.path.join(PREFIX, exp_dir_name))

    if params['task'] = 'train':
        dir_name = 'train_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.abspath(os.path.join(expdir, dir_name))

        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

        train(env, params, logdir)





