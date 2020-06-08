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
from tensorboardX import SummaryWriter

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

# def SampleNearby(starts, max_step, TB, M):
#     starts = starts + []
#     if starts==[]:
#         return []
#     else:
#         while len(starts) < M:
#             st = random.sample(starts,1)[0]
#             s_len = len(st)
#             for t in range(TB):
#                 for i in range(s_len):
#                     epsilon_x = -2 * max_step * random.random() + max_step
#                     epsilon_y = -2 * max_step * random.random() + max_step
#                     st[i][0] = st[i][0] + epsilon_x
#                     st[i][1] = st[i][1] + epsilon_y
#                     if st[i][0] > 1.0:
#                         st[i][0] = 1.0 - random.random()*0.01
#                     if st[i][0] < -1.0:
#                         st[i][0] = -1.0 + random.random()*0.01
#                     if st[i][1] > 1.0:
#                         st[i][1] = 1.0 - random.random()*0.01
#                     if st[i][1] < -1.0:
#                         st[i][1] = -1.0 + random.random()*0.01
#                 starts.append(copy.deepcopy(st))
#         return starts

def SampleNearby(starts, max_step, TB, M):
    starts = starts + []
    len_start = len(starts)
    starts_new = []
    if starts==[]:
        return []
    else:
        for i in range(len_start):
            st = copy.deepcopy(starts[i])
            s_len = len(st)
            for t in range(TB):
                for i in range(s_len):
                    epsilon_x = -2 * max_step * random.random() + max_step
                    epsilon_y = -2 * max_step * random.random() + max_step
                    st[i][0] = st[i][0] + epsilon_x
                    st[i][1] = st[i][1] + epsilon_y
                    if st[i][0] > 1.0:
                        st[i][0] = 1.0 - random.random()*0.01
                    if st[i][0] < -1.0:
                        st[i][0] = -1.0 + random.random()*0.01
                    if st[i][1] > 1.0:
                        st[i][1] = 1.0 - random.random()*0.01
                    if st[i][1] < -1.0:
                        st[i][1] = -1.0 + random.random()*0.01
                starts_new.append(copy.deepcopy(st))
                # starts.append(st)
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
    writer = SummaryWriter("/home/chenjy/new_version/logs"+args.model_dir+"/validation")

    #envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                    args.gamma, args.log_dir, device, False)

    envs = make_parallel_env(args.env_name, args.num_processes, args.seed, True)
    # env_test = make_env("simple_spread", discrete_action=True)

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
    # 给定初始的状态分布
    starts = []
    select_starts = []
    starts_old = []
    buffer_length = args.num_processes
    curri = 0
    # N_new = 1000
    N_new = 1000 #每次扩展最多能放进archive的是多少个点
    # N_old = 650
    max_step = 0.1
    TB = 3
    M = 5 * args.num_processes 
    Rmin = 0.2
    Rmax = 0.6
    Cmin = 0.4
    Cmax = 0.8
    fix_iter = 10 # 保证经过fix_iter之后，再向外扩
    count_fix = 0 # 在fixed集合上train了几个回合
    reproduce_flag = 0
    eval_iter = 5
    curri = 0
    one_starts_landmark = []
    one_starts_agent = []
    for j in range(args.num_processes):
        for i in range(now_agent_num):
            landmark_location = np.random.uniform(-1, +1, 2) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        index_sample = BatchSampler(SubsetRandomSampler(range(now_agent_num)),now_agent_num,drop_last=True)
        for indices in index_sample:
            for k in indices:
                one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+0.01))
        # pdb.set_trace()
        # select_starts.append(one_starts_agent+one_starts_landmark)
        starts_old.append(one_starts_agent+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
    # pdb.set_trace()
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    Rew = []
    

    for j in range(num_updates):
        print('count_fix: ', count_fix)
        if reproduce_flag:
            count_fix = 0
            starts = copy.deepcopy(parent_starts) #此时的select_starts暂时定为，训练环节最后一个选出来的点(简单点)
            newsample_starts = SampleNearby(starts, max_step, TB, M)
            print("newsample: ",len(newsample_starts))
            #需要测试逐个是否满足要求，然后把满足要求的加入到archive中
            curri = 0
            add_starts = []
            good_children_starts = []
            while curri < len(newsample_starts) and newsample_starts != []:
                if len(newsample_starts) - curri < args.num_processes:
                    starts = copy.deepcopy(newsample_starts[curri: len(newsample_starts)])
                    now_num_processes = len(newsample_starts)-curri
                    obs = envs.new_starts_obs(starts, now_agent_num, now_num_processes)
                    curri = len(newsample_starts)
                else:
                    starts = copy.deepcopy(newsample_starts[curri: curri + args.num_processes])
                    now_num_processes = args.num_processes
                    obs = envs.new_starts_obs(starts, now_agent_num, now_num_processes)
                    curri += args.num_processes
                rollouts = []
                for i in range(now_agent_num):
                    rollout = RolloutStorage(args.num_steps, now_num_processes,
                                            envs.observation_space[0].shape, envs.action_space[0],
                                            actor_critic[i].recurrent_hidden_state_size,
                                            now_agent_num, i, args.assign_id)
                    rollouts.append(rollout)
                # obs = envs.reset()
                if args.assign_id:
                    for i in range(now_agent_num):    
                        vec_id = np.zeros((now_num_processes, now_agent_num))
                        vec_id[:, i] = 1
                        vec_id = torch.tensor(vec_id)
                        as_obs = torch.tensor(obs.reshape(now_num_processes, -1))
                        a_obs = torch.tensor(obs[:,i,:])
                        rollouts[i].share_obs[0].copy_(torch.cat((as_obs, vec_id),1))
                        rollouts[i].obs[0].copy_(torch.cat((a_obs, vec_id),1))
                        rollouts[i].to(device)

                else:
                    for i in range(now_agent_num):
                        # pdb.set_trace()
                        rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(now_num_processes, -1)))
                        rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                        rollouts[i].to(device)
                #pdb.set_trace()
                # 保证参与训练的所有agent参数一致
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
                            value_list.append(value)
                            action_list.append(action)
                            action_log_prob_list.append(action_log_prob)
                            recurrent_hidden_states_list.append(recurrent_hidden_states)
                    # Obser reward and next obs
                    action = []
                    for i in range(now_num_processes):
                        one_env_action = []
                        for k in range(now_agent_num):
                            one_hot_action = np.zeros(envs.action_space[0].n)
                            one_hot_action[action_list[k][i]] = 1
                            one_env_action.append(one_hot_action)
                        action.append(one_env_action)
                    obs, reward, done, infos = envs.step(action, now_num_processes)

                    # for info in infos:
                    #     if 'episode' in info.keys():
                    #         episode_rewards.append(info['episode']['r'])

                    # If done then clean the history of observations.
                    '''
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done[0]])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0]
                        for info in infos[0]])
                    '''
                    masks = torch.ones(now_num_processes, 1)
                    bad_masks = torch.ones(now_num_processes, 1)
                    '''
                    rollouts.insert(obs, recurrent_hidden_states, action,
                                    action_log_prob, value, reward, masks, bad_masks)
                    '''
                    #import pdb; pdb.set_trace()
                    if args.assign_id:
                        for i in range(now_agent_num):
                            vec_id = np.zeros((now_num_processes, now_agent_num))
                            vec_id[:, i] = 1
                            vec_id = torch.tensor(vec_id)
                            as_obs = torch.tensor(obs.reshape(now_num_processes, -1))
                            a_obs = torch.tensor(obs[:,i,:])
                            rollouts[i].insert(torch.cat((as_obs, vec_id),1), torch.cat((a_obs, vec_id),1), 
                                        recurrent_hidden_states, action_list[i],
                                        action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
                    else:
                        for i in range(now_agent_num):
                            rollouts[i].insert(torch.tensor(obs.reshape(now_num_processes, -1)), torch.tensor(obs[:,i,:]), 
                                        recurrent_hidden_states, action_list[i],
                                        action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
                for i in range(len(infos)):# 这里需要换成后五帧的信息
                    if infos[i][0] < Cmax and infos[i][0] > Cmin:
                        good_children_starts.append(copy.deepcopy(starts[i]))
            if len(good_children_starts) > N_new:
                add_starts = random.sample(good_children_starts, N_new)
            else:
                add_starts = copy.deepcopy(good_children_starts)
            print("add_num: ",len(add_starts))
            if len(starts_old) + len(add_starts) <= buffer_length:
                starts_old = starts_old + add_starts
                curri = len(starts_old)-1
            else:
                starts_old = starts_old + add_starts
                starts_old = starts_old[len(starts_old)-buffer_length:]
            print('old_length: ', len(starts_old))
            reproduce_flag = 0
        else:
            starts = []
            curri_train = 0
            cover_info_list = []
            now_num_processes_train = min(args.num_processes,len(starts_old))
            obs = envs.new_starts_obs(starts_old, now_agent_num, now_num_processes_train)
            rollouts = []
            for i in range(now_agent_num):
                rollout = RolloutStorage(args.num_steps, now_num_processes_train,
                                        envs.observation_space[0].shape, envs.action_space[0],
                                        actor_critic[i].recurrent_hidden_state_size,
                                        now_agent_num, i, args.assign_id)
                rollouts.append(rollout)
        
            # obs = envs.reset()
            if args.assign_id:
                for i in range(now_agent_num):    
                    vec_id = np.zeros((now_num_processes_train, now_agent_num))
                    vec_id[:, i] = 1
                    vec_id = torch.tensor(vec_id)
                    as_obs = torch.tensor(obs.reshape(now_num_processes_train, -1))
                    a_obs = torch.tensor(obs[:,i,:])
                    rollouts[i].share_obs[0].copy_(torch.cat((as_obs, vec_id),1))
                    rollouts[i].obs[0].copy_(torch.cat((a_obs, vec_id),1))
                    rollouts[i].to(device)

            else:
                for i in range(now_agent_num):
                    # pdb.set_trace()
                    rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(now_num_processes_train, -1)))
                    rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                    rollouts[i].to(device)
            #pdb.set_trace()
            # 保证参与训练的所有agent参数一致
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
                        value_list.append(value)
                        action_list.append(action)
                        action_log_prob_list.append(action_log_prob)
                        recurrent_hidden_states_list.append(recurrent_hidden_states)
                # Obser reward and next obs
                action = []
                for i in range(now_num_processes_train):
                    one_env_action = []
                    for k in range(now_agent_num):
                        one_hot_action = np.zeros(envs.action_space[0].n)
                        one_hot_action[action_list[k][i]] = 1
                        one_env_action.append(one_hot_action)
                    action.append(one_env_action)
                obs, reward, done, infos = envs.step(action, now_num_processes_train)

                # for info in infos:
                #     if 'episode' in info.keys():
                #         episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                '''
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done[0]])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos[0]])
                '''
                masks = torch.ones(now_num_processes_train, 1)
                bad_masks = torch.ones(now_num_processes_train, 1)
                '''
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)
                '''
                #import pdb; pdb.set_trace()
                if args.assign_id:
                    for i in range(now_agent_num):
                        vec_id = np.zeros((now_num_processes_train, now_agent_num))
                        vec_id[:, i] = 1
                        vec_id = torch.tensor(vec_id)
                        as_obs = torch.tensor(obs.reshape(now_num_processes_train, -1))
                        a_obs = torch.tensor(obs[:,i,:])
                        rollouts[i].insert(torch.cat((as_obs, vec_id),1), torch.cat((a_obs, vec_id),1), 
                                    recurrent_hidden_states, action_list[i],
                                    action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
                else:
                    for i in range(now_agent_num):
                        rollouts[i].insert(torch.tensor(obs.reshape(now_num_processes_train, -1)), torch.tensor(obs[:,i,:]), 
                                    recurrent_hidden_states, action_list[i],
                                    action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)                                   
            writer.add_scalars('agent0/cover_rate',
                        {'cover_rate': np.mean(infos)},j)
            for i in range(len(infos)):
                cover_info_list.append(infos[i][0]) 

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
            # 当满足fix_iter时，把最后一次满足parent条件的点取出
            print("cover_info_list: ", len(cover_info_list))
            if count_fix == fix_iter:
                count_fix = 0
                reproduce_flag = 1
                parent_starts = []
                easy_count = 0
                del_num = 0
                for i in range(len(cover_info_list)):
                    if cover_info_list[i]> Cmax:
                        # print("del: ", starts_old[i-del_num])
                        parent_starts.append(copy.deepcopy(starts_old[i-del_num]))
                        del starts_old[i-del_num]
                        del_num += 1
                print('parent_num: ', len(parent_starts))
            else:
                count_fix += 1
            print('##############################')


        
        # validation
        # uniform init
        # Rew = []
        # # index_begin = random.sample(range(len(starts_begin)), 1)
        # obs_test = env_test.reset(now_agent_num)
        # for step in range(args.num_steps):
        #     state_test = torch.tensor([state for state in obs_test], dtype=torch.float).cuda()
        #     share_obs_test = state_test.view(1,-1)
        #     actions_test = []
        #     for i in range(now_agent_num):
        #         one_obs_test = state_test[i].view(-1, env_test.observation_space[0].shape[0])
        #         # value, action, _, recurrent_hidden_states, alpha_agent, alpha_landmark = agents[i].act(share_obs, obs, nagents, i, recurrent_hidden_states, masks)
        #         value, action, action_prob, recurrent_hidden_states = actor_critic[i].act(share_obs_test, one_obs_test, now_agent_num, recurrent_hidden_states, masks, False)
        #         # pdb.set_trace()
        #         actions_test.append(action)
        #         # al_agent.append(alpha_agent)
        #         # al_landmark.append(alpha_landmark)
        #     prob_actions = [ac.data.cpu().numpy().flatten() for ac in actions_test]
        #     actions_one_hot = []
        #     for a in prob_actions:
        #         #index = np.argmax(a)
        #         ac = np.zeros(env_test.action_space[0].n)
        #         ac[a] = 1
        #         actions_one_hot.append(ac)
        #     obs_test, rewards_test, dones_test, infos_test = env_test.step(actions_one_hot)
        #     # Rew.append(rewards_test)
        # writer.add_scalars('agent0/cover_rate_validation',
        #             {'cover_rate': np.mean(infos_test)},j)

        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            if not os.path.exists(save_path + args.model_dir):
                os.makedirs(save_path + args.model_dir)
            # 存储agent数最大值的参数
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
