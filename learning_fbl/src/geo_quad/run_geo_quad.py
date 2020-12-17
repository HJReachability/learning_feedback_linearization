
import fbl_core
from geo_quad_fbl_class import TwoControllerQuadFBL
from geo_quad_dynamics import GeoQuadDynamics 
from geo_quad_reference_generator import McClamrochCorkskrew, Lissajous, SetPoint, RandomMotions #, ConstantTwistQuadTrajectory
from pos_att_controller import PosAttController 
from fbl_core.controller import PD 
from animate_quad import QuadAnimator
import plot_functions as qplt

import numpy as np

import os
import inspect
import pickle
import time
import matplotlib.pyplot as plt

import pdb


PREFIX = os.path.abspath(os.path.join(os.path.dirname(inspect.getfile(fbl_core)), os.pardir, os.pardir, 'data'))



def fill_parameters():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['baseline', 'train', 'test'], default='baseline')
    parser.add_argument('--exp_name', type=str, default='todo')
    args = parser.parse_args()
    params = vars(args)

    params['testpath'] = '/home/valmik/learning_fbl_venv/learning_feedback_linearization/learning_fbl/data/geo_quad_different_dynamics/train_14-12-2020_16-05-27/data'

    # Description
    params['env_type'] = 'TwoControllerQuadFBL'
    params['controller_type'] = 'PosAttController'

    # Dynamics
    params['dt'] = 0.01

    # fdcl_gwo/uav_geometric_control.git matlab   
    # params['nominal_m'] = 2
    # params['nominal_J'] = [0.02, 0.02, 0.04]
    # params['nominal_ls'] = [0.315, 0.315, 0.315, 0.315]
    # params['nominal_ctfs'] = [8.004e-4, 8.004e-4, 8.004e-4, 8.004e-4]
    # params['nominal_max_us'] = [1e6, 1e6, 1e6, 1e6]
    # params['nominal_min_us'] = [-1e6, -1e6, -1e6, -1e6]

    # McClamroch Paper
    params['nominal_m'] = 4.34
    params['nominal_J'] = [0.0820, 0.0845, 0.1377]
    params['nominal_ls'] = [0.315, 0.315, 0.315, 0.315]
    params['nominal_ctfs'] = [8.004e-4, 8.004e-4, 8.004e-4, 8.004e-4]
    params['nominal_max_us'] = [1e6, 1e6, 1e6, 1e6]
    params['nominal_min_us'] = [-1e6, -1e6, -1e6, -1e6]


    params['true_m'] = 4.0
    params['true_J'] = [0.067, 0.063, 1.24]
    params['true_ls'] = [0.3145, 0.316, 0.315, 0.313]
    params['true_ctfs'] = [8e-4, 8.2e-4, 7.91e-4, 8.002e-4]
    params['true_max_us'] = [1e6, 1e6, 1e6, 1e6]
    params['true_min_us'] = [-1e6, -1e6, -1e6, -1e6]


    # params['true_m'] = params['nominal_m']
    # params['true_J'] = params['nominal_J']
    # params['true_ls'] = params['nominal_ls']
    # params['true_ctfs'] = params['nominal_ctfs']
    # params['true_max_us'] = params['nominal_max_us']
    # params['true_min_us'] = params['nominal_min_us']

    # Controller Params
    params['controller_description'] = 'two PD controllers'

    # fdcl_gwo/uav_geometric_control.git matlab
    # params['pos_p_gain'] = 10
    # params['pos_v_gain'] = 8
    # params['att_p_gain'] = 1.5
    # params['att_v_gain'] = 0.35

    # McClamroch Paper
    params['pos_p_gain'] = 16*4.34
    params['pos_v_gain'] = 5.6*4.34
    params['att_p_gain'] = 8.81
    params['att_v_gain'] = 2.54

    # Reference
    # params['ref_type'] = 'constant_twist'
    # params['ref_type'] = 'mcclamroch_corkscrew'
    params['ref_type'] = 'lissajous'
    # params['ref_type'] = 'setpoint'
    # params['ref_type'] = 'random_motions'
    params['nominal_traj_length'] = 100

    # params['test_ref_type'] = 'lissajous'

    # FBL class
    params['action_scaling'] = 1
    params['reward_scaling'] = 1
    params['reward_norm'] = 1

    # State Limits
    params['init_upper_limits'] = np.array([8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5])
    params['init_lower_limits'] = -np.array([8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5])

    params['obs_upper_limits'] = 1.05*np.array([1e2,1e2,1e2,1,1,1,1,1,1,1,1,1,5e4,5e4,5e4,5e6,5e6,5e6])
    params['obs_lower_limits'] = -1.05*np.array([1e2,1e2,1e2,1,1,1,1,1,1,1,1,1,5e4,5e4,5e4,5e6,5e6,5e6])

    # params['obs_upper_limits'] = np.array([10,10,10,1,1,1,1,1,1,1,1,1,50,50,50,500,500,500])
    # params['obs_lower_limits'] = -np.array([10,10,10,1,1,1,1,1,1,1,1,1,50,50,50,500,500,500])


    params['action_upper_limits'] = 1
    params['action_lower_limits'] = -1

    # Learning
    params['learning_alg'] = 'ppo'
    params['learning_rate'] = 3e-4
    params['n_steps'] = 2048
    params['batch_size'] = 64
    params['n_epochs'] = 10
    params['gamma'] = 0.99
    params['gae_lambda'] = 0.95
    params['total_timesteps'] = 3e5
    params['shared_layers'] = [256, 256]
    params['vf_layers'] = []
    params['pi_layers'] = []
    params['log_std_init'] = np.log(1e-3)


    return params


def create_environment(params):

    import gym
    from gym import spaces
    import learning_fbl_env
    
    ############ Create Dynamics

    nominal_J = np.diag(params['nominal_J'])
    true_J = np.diag(params['true_J'])

    nominal_dynamics = GeoQuadDynamics(params['nominal_m'], nominal_J, params['nominal_ls'], params['nominal_ctfs'], params['nominal_max_us'], params['nominal_min_us'], time_step = params['dt'])
    true_dynamics = GeoQuadDynamics(params['true_m'], true_J, params['true_ls'], params['true_ctfs'], params['true_max_us'], params['true_min_us'], time_step = params['dt'])

    ############# Create Controller
    pos_controller = PD(3,3,params['pos_p_gain'], params['pos_v_gain'])
    att_controller = PD(3,3,params['att_p_gain'], params['att_v_gain'])
    controller = PosAttController(pos_controller, att_controller)

    ############# Create Reference
    if params['ref_type'] == 'constant_twist':
        reference_generator = ConstantTwistQuadTrajectory(params['init_upper_limits'], params['init_lower_limits'], 
                                                          nominal_dynamics, params['nominal_traj_length'])
    elif params['ref_type'] == 'mcclamroch_corkscrew':
        reference_generator = McClamrochCorkskrew(params['init_upper_limits'], params['init_lower_limits'], 
                                                  nominal_dynamics)
    elif params['ref_type'] == 'lissajous':
        reference_generator = Lissajous(params['init_upper_limits'], params['init_lower_limits'], 
                                        nominal_dynamics)
    elif params['ref_type'] == 'setpoint':
        reference_generator = SetPoint(params['init_upper_limits'], params['init_lower_limits'], 
                                       nominal_dynamics)
    elif params['ref_type'] == 'random_motions':
        reference_generator = RandomMotions(params['init_upper_limits'], params['init_lower_limits'], 
                                            nominal_dynamics, params['nominal_traj_length'])


    ############ Test ref

    # if params['test_ref_type'] == 'constant_twist':
    #     test_reference_generator = ConstantTwistQuadTrajectory(params['init_upper_limits'], params['init_lower_limits'], 
    #                                                       nominal_dynamics, params['nominal_traj_length'])
    # elif params['test_ref_type'] == 'mcclamroch_corkscrew':
    #     test_reference_generator = McClamrochCorkskrew(params['init_upper_limits'], params['init_lower_limits'], 
    #                                               nominal_dynamics)
    # elif params['test_ref_type'] == 'lissajous':
    #     test_reference_generator = Lissajous(params['init_upper_limits'], params['init_lower_limits'], 
    #                                     nominal_dynamics)
    # elif params['test_ref_type'] == 'setpoint':
    #     test_reference_generator = SetPoint(params['init_upper_limits'], params['init_lower_limits'], 
    #                                    nominal_dynamics)
    # elif params['test_ref_type'] == 'random_motions':
    #     test_reference_generator = RandomMotions(params['init_upper_limits'], params['init_lower_limits'], 
    #                                         nominal_dynamics, params['nominal_traj_length'])


    ########### Create action and observation spaces
    udim = nominal_dynamics.udim
    xdim = nominal_dynamics.xdim

    action_space = spaces.Box(low=params['action_lower_limits'], high=params['action_upper_limits'], shape=(udim**2 + udim,), dtype=np.float32)
    observation_space = spaces.Box(low=params['obs_lower_limits'], high=params['obs_upper_limits'], shape=(xdim,), dtype=np.float32)


    ########## Create FBL Class
    fbl_obj = TwoControllerQuadFBL(action_space = action_space, observation_space = observation_space,
        nominal_dynamics = nominal_dynamics, true_dynamics = true_dynamics,
        cascaded_quad_controller = controller,
        reference_generator = reference_generator,
        action_scaling = params['action_scaling'], reward_scaling = params['reward_scaling'], reward_norm = params['reward_norm'])

    ########## Test Class
    # test_fbl_obj = TwoControllerQuadFBL(action_space = action_space, observation_space = observation_space,
    #     nominal_dynamics = nominal_dynamics, true_dynamics = true_dynamics,
    #     cascaded_quad_controller = controller,
    #     reference_generator = test_reference_generator,
    #     action_scaling = params['action_scaling'], reward_scaling = params['reward_scaling'], reward_norm = params['reward_norm'])


    # Create environment
    # env = lambda : gym.make('learning_fbl_env:LearningFBLEnv-v0', learn_fbl = fbl_obj)
    env = gym.make('learning_fbl_env:LearningFBLEnv-v0', learn_fbl = fbl_obj)
    # test_env = gym.make('learning_fbl_env:LearningFBLEnv-v0', learn_fbl = test_fbl_obj)

    return env


def train(env, params, logdir):

    # pdb.set_trace()

    if params['learning_alg'] == 'ppo':
        from stable_baselines3 import PPO
        from stable_baselines3.ppo import MlpPolicy

        net_arch = params['shared_layers'] + [dict(vf=params['vf_layers'], pi=params['pi_layers'])]
        policy_kwargs = dict(net_arch=net_arch, log_std_init=params['log_std_init'])

        tensorboard_logdir = logdir
        model = PPO(MlpPolicy, env, verbose = 1,
                learning_rate = params['learning_rate'],
                n_steps = params['n_steps'],
                batch_size = params['batch_size'],
                n_epochs = params['n_epochs'],
                gamma = params['gamma'],
                gae_lambda = params['gae_lambda'],
                tensorboard_log = logdir,
                policy_kwargs = policy_kwargs
            )
    
    with open(logdir + '/params.pkl', 'wb') as f:
        pickle.dump(params, f, 0)

    model.learn(total_timesteps = params['total_timesteps'])
    savefile = logdir + '/data'
    model.save(savefile)

def test(env, params, logdir):

    # pdb.set_trace()
    with open(logdir + '/params.pkl', 'wb') as f:
        pickle.dump(params, f, 0)

    if params['learning_alg'] == 'ppo':
        from stable_baselines3 import PPO
        from stable_baselines3.ppo import MlpPolicy

        # net_arch = params['shared_layers'] + [dict(vf=params['vf_layers'], pi=params['pi_layers'])]
        # policy_kwargs = dict(net_arch=net_arch, log_std_init=params['log_std_init'])

        # tensorboard_logdir = logdir
        # model = PPO(MlpPolicy, env, verbose = 1,
        #         learning_rate = params['learning_rate'],
        #         n_steps = params['n_steps'],
        #         batch_size = params['batch_size'],
        #         n_epochs = params['n_epochs'],
        #         gamma = params['gamma'],
        #         gae_lambda = params['gae_lambda'],
        #         tensorboard_log = logdir,
        #         policy_kwargs = policy_kwargs
        #     )

        model = PPO.load(params['testpath'])

        rollouts = []

        dt = env.learn_fbl._nominal_dynamics.dt
        animator = QuadAnimator(env.learn_fbl.observation_space.low[:3], 
                                env.learn_fbl.observation_space.high[:3],
                                dt)

        for i in range(1):
            done = False
            x = env.unnormalize_observation(env.reset())
            ref = env.learn_fbl._reference

            animator.start_animation(x, ref)

            states = [x]
            rewards = []
            xs = [x[0]]
            ys = [x[1]]
            zs = [x[2]]

            max_T = len(ref)*dt
            frame_count = int(max_T // 0.05)
            rate = int(len(ref)//frame_count)

            i = 0
            while not done:
                # pdb.set_trace()
                print(x)
                a = model.predict(env.normalize_observation(x))
                x, r, done, _ = env.step(a[0])
                x = env.unnormalize_observation(x)

                i = i+1

                if i == rate:
                    animator.update_animation(x)
                    i = 0

                xs.append(x[0])
                ys.append(x[1])
                zs.append(x[2])
                states.append(x)
                rewards.append(r)

            animator.update_animation(x)

            animator.plot_line(xs, ys, zs)
            plt.pause(1)
            rollouts.append((states, rewards))

            # import pdb 
            # pdb.set_trace()

            qplt.plot_state_trajectory(states, ref, dt)

        
        print(np.sum(np.array(rewards)))

        with open(logdir + 'rollout_data.pkl', 'wb') as f:
            pickle.dump(rollouts, f, 0)




def baseline(env, params, logdir):

    with open(logdir + '/params.pkl', 'wb') as f:
        pickle.dump(params, f, 0)

    a = np.zeros(20,)
    # D = np.reshape(np.eye(4), (16,))
    # h = np.zeros(4,)
    # a = np.concatenate([D, h])
    rollouts = []

    dt = env.learn_fbl._nominal_dynamics.dt
    animator = QuadAnimator(env.learn_fbl.observation_space.low[:3], 
                            env.learn_fbl.observation_space.high[:3],
                            dt)

    for i in range(1):
        done = False
        x = env.unnormalize_observation(env.reset())
        ref = env.learn_fbl._reference

        animator.start_animation(x, ref)

        states = [x]
        rewards = []
        xs = [x[0]]
        ys = [x[1]]
        zs = [x[2]]

        max_T = len(ref)*dt
        frame_count = int(max_T // 0.05)
        rate = int(len(ref)//frame_count)

        i = 0
        while not done:
            x, r, done, _ = env.step(env.normalize_action(a))
            x = env.unnormalize_observation(x)

            i = i+1

            # done = (i == len(ref))

            if i == rate:
                animator.update_animation(x)
                i = 0

            xs.append(x[0])
            ys.append(x[1])
            zs.append(x[2])
            states.append(x)
            rewards.append(r)

        animator.update_animation(x)

        animator.plot_line(xs, ys, zs)
        plt.pause(1)
        rollouts.append((states, rewards))
        plt.show()

        # import pdb 
        # pdb.set_trace()

        qplt.plot_state_trajectory(states, ref, dt)



    
    print(np.sum(np.array(rewards)))

    with open(logdir + '/rollout_data.pkl', 'wb') as f:
        pickle.dump(rollouts, f, 0)





if __name__ == '__main__':

    params = fill_parameters()
    env = create_environment(params)

    exp_dir_name = 'geo_quad_' + params['exp_name']
    expdir = os.path.abspath(os.path.join(PREFIX, exp_dir_name))

    if params['task'] == 'train':
        dir_name = 'train_' + time.strftime("%d-%m-%Y_%H-%M-%S") + '/'
        logdir = os.path.abspath(os.path.join(expdir, dir_name))

        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

        train(env, params, logdir)

    if params['task'] == 'baseline':
        dir_name = 'baseline_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.abspath(os.path.join(expdir, dir_name))

        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

        baseline(env, params, logdir)

    if params['task'] == 'test':
        dir_name = 'test_' + time.strftime("%d-%m-%Y_%H-%M-%S") + '/'
        logdir = os.path.abspath(os.path.join(expdir, dir_name))

        if not (os.path.exists(logdir)):
            os.makedirs(logdir)

        print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

        test(env, params, logdir)





