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
from a2c_ppo_acktr.model import Policy2, ATTBase2
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
                    # pdb.set_trace()
                    # tmp_epsilon_x = max_step * random.random() #0-0.01
                    # tmp_epsilon_y = max_step * random.random()
                    # if st[i][0] > 1.0:
                    #     st[i][0] -= tmp_epsilon_x
                    # if st[i][0] < -1.0:
                    #     st[i][0] += tmp_epsilon_x
                    # if st[i][1] > 1.0:
                    #     st[i][1] -= tmp_epsilon_y
                    # if st[i][1] < -1.0:
                    #     st[i][1] += tmp_epsilon_y
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
        if len(starts_new) <= N_new:
            starts_new = random.sample(starts_new, len(starts_new))
        else:
            starts_new = random.sample(starts_new, N_new)
        return starts_new


def main():
    args = get_args()
    model_dir = args.model_dir + '_' + str(args.adv_num) + str(args.good_num) + str(args.landmark_num)
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
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = make_parallel_env(args.env_name, args.num_processes, args.seed, True)
    actor_critic = []
    now_agent_num = args.adv_num
    if args.share_policy:
        if args.use_attention:
            share_base = ATTBase2(envs.observation_space[0].shape[0], hidden_size=100)
            share_dist = Categorical(share_base.output_size, envs.action_space[0].n)
            for i in range(args.adv_num):
                ac = Policy2(
                    envs.observation_space[0].shape,
                    envs.action_space[0],
                    agent_num=args.adv_num, 
                    agent_i=i,
                    base=share_base,
                    dist=share_dist,
                    base_kwargs={'recurrent': args.recurrent_policy},
                    adv_num=args.adv_num, 
                    good_num=args.good_num,
                    landmark_num=args.landmark_num)
                ac.to(device)
                actor_critic.append(ac)
        else:
            ac = Policy2(
                    envs.observation_space[0].shape,
                    envs.action_space[0],
                    agent_num=args.adv_num, 
                    agent_i=0,
                    base_kwargs={'recurrent': args.recurrent_policy, 'assign_id': args.assign_id})
            ac.to(device)
            for i in range(args.adv_num):
                actor_critic.append(ac)
    else:
        for i in range(args.adv_num):
            ac = Policy2(
                envs.observation_space[0].shape,
                envs.action_space[0],
                agent_num=args.adv_num, 
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
        agent = []
        for i in range(args.adv_num):
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
                model_dir = model_dir))
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
        
    episode_rewards = deque(maxlen=10)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    ratio = 1
    # 给定初始的状态分布
    starts = []
    select_starts = []
    starts_old = []
    buffer_length = 10000
    curri = 0
    # N_new = 1000
    N_new = 50
    N_old = 650
    max_step = 0.1
    TB = 1
    M = 10000  
    Rmin = 0.2
    Rmax = 0.6
    Cmin = 0.0
    Cmax = 0.8
    fix_iter = 50 # 保证经过fix_iter之后，再向外扩
    count_fix = 0 # 在fixed集合上train了几个回合
    proportion = 2.0 # >1时代表N_new和N_old比例不变
    one_starts_landmark = []
    one_starts_good_agent = []
    one_starts_adv_agent = []
    for j in range(args.num_processes):
        for i in range(args.landmark_num):
            landmark_location = np.random.uniform(-1, +1, 2) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        index_sample = BatchSampler(SubsetRandomSampler(range(args.landmark_num)),args.landmark_num, drop_last=True)
        for indices in index_sample:
            for k in indices:
                one_starts_good_agent.append(copy.deepcopy(one_starts_landmark[k]+0.01))
        for i in range(args.adv_num):
            # 两个agent分别在球的两端
            if i % 2 == 0:
                one_starts_adv_agent.append(copy.deepcopy(one_starts_good_agent[int(i/2)]+0.25))
            else:
                one_starts_adv_agent.append(copy.deepcopy(one_starts_good_agent[int(i/2)]-0.25))
        # pdb.set_trace()
        select_starts.append(one_starts_adv_agent + one_starts_good_agent + one_starts_landmark)
        starts_old.append(one_starts_adv_agent + one_starts_good_agent + one_starts_landmark)
        one_starts_good_agent = []
        one_starts_adv_agent = []
        one_starts_landmark = []
    

    # samplenearby需要改，初始状态的设置需要改，new_starts_obs需要改
    for j in range(num_updates):
        #pdb.set_trace()
        if count_fix == fix_iter:
            starts = copy.deepcopy(select_starts)
            newsample_starts = SampleNearby(starts, N_new, max_step, TB, M)
            # N_old保持小于args.num_processes
            index_old = random.sample(range(len(starts_old)),N_old)
            if len(newsample_starts)< args.num_processes-len(index_old): # 说明扩展出来的很少/select的很少
                index_new = random.sample(range(len(newsample_starts)), len(newsample_starts))
                index_old = random.sample(range(len(starts_old)), args.num_processes-len(index_new))
            else:
                index_new = random.sample(range(len(newsample_starts)), args.num_processes-len(index_old))
            # index_old.sort()
            starts = []
            print('new_node: ', len(index_new))
            print('old_node: ', len(index_old))
            for i in range(len(index_new)):
                starts.append(copy.deepcopy(newsample_starts[index_new[i]]))
            for i in range(len(index_old)):
                starts.append(copy.deepcopy(starts_old[index_old[i]]))
        else:
            starts = []
            index_old = random.sample(range(len(starts_old)), args.num_processes)
            for i in range(len(index_old)):
                starts.append(copy.deepcopy(starts_old[index_old[i]]))
        obs = envs.new_starts_obs(starts, now_agent_num)

        # obs = envs.reset(now_agent_num)  #[num_process[n_agents[obs_dim]]]
        rollouts = []
        for i in range(args.adv_num):
            rollout = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space[0].shape, envs.action_space[0],
                              actor_critic[i].recurrent_hidden_state_size,
                              args.adv_num, i, args.assign_id)
            rollouts.append(rollout)
        if args.assign_id:
            for i in range(args.adv_num):    
                vec_id = np.zeros((args.num_processes, args.adv_num))
                vec_id[:, i] = 1
                vec_id = torch.tensor(vec_id)
                as_obs = torch.tensor(obs.reshape(args.num_processes, -1))
                a_obs = torch.tensor(obs[:,i,:])
                rollouts[i].share_obs[0].copy_(torch.cat((as_obs, vec_id),1))
                rollouts[i].obs[0].copy_(torch.cat((a_obs, vec_id),1))
                rollouts[i].to(device)
        else:
            for i in range(args.adv_num):
                rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.num_processes, -1)))
                rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                rollouts[i].to(device)

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            for i in range(args.adv_num):
                utils.update_linear_schedule(
                    agent[i].optimizer, j, num_updates,
                    agent[i].optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            value_list, action_list, action_log_prob_list, recurrent_hidden_states_list = [], [], [], []
            with torch.no_grad():
                for i in range(args.adv_num):
                    #pdb.set_trace()
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic[i].act(
                        rollouts[i].share_obs[step],
                        rollouts[i].obs[step], args.adv_num, rollouts[i].recurrent_hidden_states[step],
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
                for k in range(args.adv_num):
                    one_hot_action = np.zeros(envs.action_space[0].n)
                    one_hot_action[action_list[k][i]] = 1
                    one_env_action.append(one_hot_action)
                action.append(one_env_action)
            #start = time.time()
            obs, reward, done, infos = envs.step(action)
            #print(step,reward)

            #print(reward.max())
            #if(reward.max()>2):
            #    import pdb; pdb.set_trace()
            #end = time.time()
            #print("step time: ", end-start)
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
            masks = torch.ones(args.num_processes, 1)
            bad_masks = torch.ones(args.num_processes, 1)
            '''
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            '''
            #import pdb; pdb.set_trace()
            if args.assign_id:
                for i in range(args.adv_num):
                    vec_id = np.zeros((args.num_processes, args.adv_num))
                    vec_id[:, i] = 1
                    vec_id = torch.tensor(vec_id)
                    as_obs = torch.tensor(obs.reshape(args.num_processes, -1))
                    a_obs = torch.tensor(obs[:,i,:])
                    rollouts[i].insert(torch.cat((as_obs, vec_id),1), torch.cat((a_obs, vec_id),1), 
                                recurrent_hidden_states, action_list[i],
                                action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
            else:
                for i in range(args.adv_num):
                    rollouts[i].insert(torch.tensor(obs.reshape(args.num_processes, -1)), torch.tensor(obs[:,i,:]), 
                                recurrent_hidden_states, action_list[i],
                                action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)

        if count_fix == fix_iter:
            count_fix = 0
            select_starts = [] 
            add_starts = []
            easy_count = 0
            del_count = 0
            index_old = np.array(index_old)
            for i in range(len(infos)):
                if infos[i][0]>= Cmin and infos[i][0]<= Cmax:
                    if i < len(index_new):
                        select_starts.append(copy.deepcopy(starts[i]))
                        add_starts.append(copy.deepcopy(starts[i]))
                    else:
                        select_starts.append(copy.deepcopy(starts[i]))
                elif infos[i][0] > Cmax:
                    easy_count += 1
                    # 此处需要区分是不是从old中来的点，如果是，需要清除
                    # if i >= len(index_new):
                    #     del starts_old[index_old[i-len(index_new)]]
                    #     del_count += 1
                    #     index_old = index_old - 1 
            print('select_num: ', len(select_starts))
            print('add_num: ', len(add_starts))
            print('easy_num: ', easy_count)
            # print('del_num: ', del_count)

            if len(starts_old) < buffer_length:
                starts_old = starts_old + add_starts
                curri = len(starts_old)-1
            else:
                starts_old = starts_old[len(add_starts):] + add_starts
        else:
            count_fix += 1
            mid_count = 0
            easy_count = 0
            for i in range(len(infos)):
                if infos[i][0]>= Cmin and infos[i][0]<= Cmax:
                    mid_count += 1
                    # print("mid_reward: ", reward[i])
                    # print("mid_mean_reward: ", rollouts[0].rewards.mean(axis=0)[i])
                elif infos[i][0] > Cmax:
                    easy_count += 1
                    # print("easy_reward: ", reward[i])
                    # print("easy_mean_reward: ", rollouts[0].rewards.mean(axis=0)[i])
                    # 此处需要区分是不是从old中来的点，如果是，需要清除
                    # if i >= len(index_new):
                    #     del starts_old[index_old[i-len(index_new)]]
                    #     del_count += 1
                    #     index_old = index_old - 1 
            print('mid_num: ', mid_count)
            print('easy_num: ', easy_count)
            if count_fix == fix_iter:
                if easy_count/args.num_processes > proportion:
                    N_new = N_new + 50
                    if N_new > args.num_processes:
                        N_new = args.num_processes
                    N_old = args.num_processes - N_new
        print('count_fix: ', count_fix)
        print('old_length: ', len(starts_old))
        print('##############################')

        with torch.no_grad():
            next_value_list = []
            for i in range(args.adv_num):
                next_value = actor_critic[i].get_value(
                    rollouts[i].share_obs[-1],
                    rollouts[i].obs[-1], args.adv_num, rollouts[i].recurrent_hidden_states[-1],
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
        
        for i in range(args.adv_num):
            rollouts[i].compute_returns(next_value_list[i], args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

        for i in range(args.adv_num):
            value_loss, action_loss, dist_entropy = agent[i].update(rollouts[i], args.adv_num)
            #import pdb; pdb.set_trace()
            if (i == 0 and (j+1)%10 == 0):
                print("update num: " + str(j+1) + " value loss: " + str(value_loss))
        #rollouts.after_update()

        # save for every interval-th episode or for the last epoch     
        #pdb.set_trace()   
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            if not os.path.exists(save_path + model_dir):
                os.makedirs(save_path + model_dir)
            for i in range(args.adv_num):
                torch.save([
                    actor_critic[i],
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], save_path + model_dir + '/agent_%i' % (i+1) + ".pt")

if __name__ == "__main__":
    main()
