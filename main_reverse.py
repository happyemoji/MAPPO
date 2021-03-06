 # -*- coding: UTF-8 -*-
import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, ATTBase
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from utils.make_env import make_env                         
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import pdb
import random
import copy
import matplotlib.pyplot as plt

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#env = make_env("simple_spread", discrete_action=True)
#env.seed(1)

#torch.set_default_tensor_type('torch.DoubleTensor')

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def SampleNearby(starts, N_new, max_step, TB, M):
    starts = starts + []
    if starts==[]:
        return []
    else:
        while len(starts) < M:
            st = random.sample(starts,1)[0]
            s_len = len(st)
            for t in range(TB):
                for i in range(s_len):
                    epsilon_x = -2 * max_step * random.random() + max_step
                    epsilon_y = -2 * max_step * random.random() + max_step
                    st[i][0] = st[i][0] + epsilon_x
                    st[i][1] = st[i][1] + epsilon_y
                    # pdb.set_trace()
                    tmp_epsilon_x = max_step * random.random() #0-0.01
                    tmp_epsilon_y = max_step * random.random()
                    if st[i][0] > 1.0:
                        st[i][0] -= tmp_epsilon_x
                    if st[i][0] < -1.0:
                        st[i][0] += tmp_epsilon_x
                    if st[i][1] > 1.0:
                        st[i][1] -= tmp_epsilon_y
                    if st[i][1] < -1.0:
                        st[i][1] += tmp_epsilon_y
                starts.append(copy.deepcopy(st))
                # starts.append(st)
            # pdb.set_trace()
        starts_new = random.sample(starts,N_new)
        return starts_new




def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    #envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                    args.gamma, args.log_dir, device, False)

    envs = make_parallel_env(args.env_name, args.num_processes, args.seed, True)

    '''
    actor_critic = Policy(
        envs.observation_space[0].shape,
        envs.action_space[0],
        agent_num=args.agent_num, 
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    '''
    actor_critic = []
    # for i in range(args.agent_num):
    #     # actor_critic, ob_rms = torch.load('/home/chenjy/new_version/trained_models/ppo' + args.model_dir + '/agent_%i' % (i+1) + ".pt")
    #     ac, ob_rms = torch.load('/home/chenjy/new_version/trained_models/ppo/run4_reverse_3'  + '/agent_1' + ".pt")
    #     ac.to(device)
    #     actor_critic.append(ac)
    if args.share_policy:
        if args.use_attention:
            share_base = ATTBase(envs.observation_space[0].shape[0], args.agent_num)
            share_dist = Categorical(share_base.output_size, envs.action_space[0].n)
            for i in range(args.agent_num):
                ac = Policy(
                    envs.observation_space[0].shape,
                    envs.action_space[0],
                    agent_num=args.agent_num, 
                    agent_i=i,
                    base=share_base,
                    dist=share_dist,
                    base_kwargs={'recurrent': args.recurrent_policy})
                ac.to(device)
                actor_critic.append(ac)
        else:
            ac = Policy(
                    envs.observation_space[0].shape,
                    envs.action_space[0],
                    agent_num=args.agent_num, 
                    agent_i=0,
                    base_kwargs={'recurrent': args.recurrent_policy, 'assign_id': args.assign_id})
            ac.to(device)
            for i in range(args.agent_num):
                actor_critic.append(ac)
    else:
        for i in range(args.agent_num):
            ac = Policy(
                envs.observation_space[0].shape,
                envs.action_space[0],
                agent_num=args.agent_num, 
                agent_i=i,
                base_kwargs={'recurrent': args.recurrent_policy})
            ac.to(device)
            actor_critic.append(ac)
    #import pdb; pdb.set_trace()
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        '''
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        '''
        agent = []
        for i in range(args.agent_num):
            agent.append(algo.PPO(
                actor_critic[i],
                i,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                model_dir = args.model_dir))
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)
    '''   
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space[0].shape, envs.action_space[0],
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(torch.tensor(obs[:,0,:]))
    rollouts.to(device)
    '''

    # rollouts = []
    # for i in range(args.agent_num):
    #     rollout = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs.observation_space[0].shape, envs.action_space[0],
    #                           actor_critic[i].recurrent_hidden_state_size,
    #                           args.agent_num, i, args.assign_id)
    #     rollouts.append(rollout)

        
    episode_rewards = deque(maxlen=10)

    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    print(num_updates)

    now_agent_num = args.agent_num
    ratio = 1
    # ???????????????????????????
    starts = []
    starts_old = []
    buffer_length = 10000
    curri = 0
    N_new = 1000
    N_old = 500
    max_step = 0.2
    TB = 1
    M = 10000
    Rmin = 0.1
    Rmax = 0.7
    Cmin = 0.6
    Cmax = 0.8
    one_starts_landmark = []
    one_starts_agent = []
    for j in range(N_new):
        for i in range(now_agent_num):
            landmark_location = np.random.uniform(-1, +1, 2) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        index_sample = BatchSampler(SubsetRandomSampler(range(now_agent_num)),now_agent_num,drop_last=True)
        for indices in index_sample:
            for k in indices:
                one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+0.01))
        # pdb.set_trace()
        starts.append(one_starts_agent+one_starts_landmark)
        starts_old.append(one_starts_agent+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
    curri = len(starts_old)
    # ??????agent?????????landmark?????????????????????????????????????????????
    # pdb.set_trace()
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    
    for j in range(num_updates):
        starts = SampleNearby(starts, N_new, max_step, TB, M)
        starts = starts + random.sample(list(starts_old), N_old)
        sample_index = random.sample(range(len(starts)), args.num_processes)# ???good starts??????????????????500???????????????
        # ??????starts??????obs
        obs, rou_index= envs.new_starts_obs(starts, sample_index, now_agent_num)# ?????????starts????????????????????????
        print("now_agent_num: ", now_agent_num)
        rollouts = []
        for i in range(now_agent_num):
            rollout = RolloutStorage(args.num_steps, args.num_processes,
                                  envs.observation_space[0].shape, envs.action_space[0],
                                  actor_critic[i].recurrent_hidden_state_size,
                                  now_agent_num, i, args.assign_id)
            rollouts.append(rollout)
        # obs = envs.reset()
        if args.assign_id:
            for i in range(now_agent_num):    
                vec_id = np.zeros((args.num_processes, now_agent_num))
                vec_id[:, i] = 1
                vec_id = torch.tensor(vec_id)
                as_obs = torch.tensor(obs.reshape(args.num_processes, -1))
                a_obs = torch.tensor(obs[:,i,:])
                rollouts[i].share_obs[0].copy_(torch.cat((as_obs, vec_id),1))
                rollouts[i].obs[0].copy_(torch.cat((a_obs, vec_id),1))
                rollouts[i].to(device)

        else:
            for i in range(now_agent_num):
                # pdb.set_trace()
                rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.num_processes, -1)))
                rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                rollouts[i].to(device)
        #pdb.set_trace()
        # ???????????????????????????agent????????????
        # for i in range(now_agent_num):
        #     agent[i] = agent[0]

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            for i in range(now_agent_num):
                utils.update_linear_schedule(
                    agent[i].optimizer, j, num_updates,
                    agent[i].optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            value_list, action_list, action_log_prob_list, recurrent_hidden_states_list = [], [], [], []
            with torch.no_grad():
                for i in range(now_agent_num):
                    #pdb.set_trace()
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic[i].act(
                        rollouts[i].share_obs[step],
                        rollouts[i].obs[step], now_agent_num, rollouts[i].recurrent_hidden_states[step],
                        rollouts[i].masks[step])
                    #import pdb; pdb.set_trace()
                    value_list.append(value)
                    action_list.append(action)
                    action_log_prob_list.append(action_log_prob)
                    recurrent_hidden_states_list.append(recurrent_hidden_states)
            # Obser reward and next obs
            action = []
            for i in range(args.num_processes):
                one_env_action = []
                for k in range(now_agent_num):
                    one_hot_action = np.zeros(envs.action_space[0].n)
                    one_hot_action[action_list[k][i]] = 1
                    one_env_action.append(one_hot_action)
                action.append(one_env_action)
         
            obs, reward, done, infos = envs.step(action)


            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            '''
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done[0]])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos[0]])
            '''
            masks = torch.ones(args.num_processes, 1)
            bad_masks = torch.ones(args.num_processes, 1)
            '''
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            '''
            #import pdb; pdb.set_trace()
            if args.assign_id:
                for i in range(now_agent_num):
                    vec_id = np.zeros((args.num_processes, now_agent_num))
                    vec_id[:, i] = 1
                    vec_id = torch.tensor(vec_id)
                    as_obs = torch.tensor(obs.reshape(args.num_processes, -1))
                    a_obs = torch.tensor(obs[:,i,:])
                    rollouts[i].insert(torch.cat((as_obs, vec_id),1), torch.cat((a_obs, vec_id),1), 
                                recurrent_hidden_states, action_list[i],
                                action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
            else:
                for i in range(now_agent_num):
                    rollouts[i].insert(torch.tensor(obs.reshape(args.num_processes, -1)), torch.tensor(obs[:,i,:]), 
                                recurrent_hidden_states, action_list[i],
                                action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
        
        with torch.no_grad():
            next_value_list = []
            for i in range(now_agent_num):
                next_value = actor_critic[i].get_value(
                    rollouts[i].share_obs[-1],
                    rollouts[i].obs[-1], now_agent_num, rollouts[i].recurrent_hidden_states[-1],
                    rollouts[i].masks[-1]).detach()
                next_value_list.append(next_value)

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])
        
        for i in range(now_agent_num):
            rollouts[i].compute_returns(next_value_list[i], args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

        for i in range(now_agent_num):
            value_loss, action_loss, dist_entropy = agent[i].update(rollouts[i], now_agent_num)
            #import pdb; pdb.set_trace()
            if (i == 0 and (j+1)%10 == 0):
                print("update num: " + str(j+1) + " value loss: " + str(value_loss))
        # print('rollouts.rewards: ', rollouts[0].rewards.shape)

        # select starts using Rew
        # Rew = rollouts[0].rewards.mean(axis=0)
        # starts_tmp = [] 
        # for i in range(len(Rew)):
        #     if Rew[i]>= Rmin and Rew[i]<= Rmax:
        #         starts_tmp.append(copy.deepcopy(starts[rou_index[i]]))
        # starts = copy.deepcopy(starts_tmp)

        # select starts using coverage rate
        starts_tmp = [] 
        for i in range(len(infos)):
            if infos[i]['n'][0]>= Cmin and infos[i]['n'][0]<= Cmax:
                starts_tmp.append(copy.deepcopy(starts[rou_index[i]]))
        starts = copy.deepcopy(starts_tmp)

        if len(starts_old) < buffer_length:
            starts_old = starts_old + starts
            curri = len(starts_old)-1
        else:
            starts_old = np.roll(starts_old,len(starts))
            starts_old[0:len(starts)] = copy.deepcopy(starts)

        # if starts!=[]:
        #     if len(starts)>=1:
        #         for i in range(now_agent_num):
        #             x1.append(starts[0][i][0])
        #             y1.append(starts[0][i][1])
        #             x2.append(starts[0][i+now_agent_num][0])
        #             y2.append(starts[0][i+now_agent_num][1])
        # plt.scatter(x1,y1,c='#0000FF')
        # plt.scatter(x2,y2,c='#A52A2A')
        # x1 = []
        # y1 = []
        # x2 = []
        # y2 = []
        # if j % 1 == 0 :
        #     plt.savefig('/home/chenjy/new_version/select_agent_landmark/test_'+str(j)+'.jpg')
        #     plt.clf()

        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            if not os.path.exists(save_path + args.model_dir):
                os.makedirs(save_path + args.model_dir)
            # ??????agent?????????????????????
            for i in range(now_agent_num):
                torch.save([
                    actor_critic[i],
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], save_path + args.model_dir + '/agent_%i' % (i+1) + ".pt")
        '''
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
        '''
        '''
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
        '''

if __name__ == "__main__":
    main()
