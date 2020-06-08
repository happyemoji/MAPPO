import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from torch.autograd import Variable
import imageio
#from a2c_ppo_acktr.model import MLPBase, CNNBase
from a2c_ppo_acktr.arguments import get_args
import pdb
import time
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import copy
import random

# Parameters
gamma = 0.95
render = False
seed = 0
log_interval = 10

# def num_reach(world):
#     num = 0
#     for a in world.agents:
#         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in world.landmarks]
#         if min(dists) <= world.agents[0].size + world.landmarks[0].size:
#             num = num + 1
#     return num 
def num_reach(world):
    num = 0
    for l in world.landmarks:
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        if min(dists) <= world.agents[0].size + world.landmarks[0].size:
            num = num + 1
    return num 

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

if __name__ == '__main__':
    ### 手动加agent
    # agent1 = Actor()
    # agent2 = Actor()
    # agent3 = Actor()
    # agent4 = Actor()
    # agent5 = Actor()
    args = get_args()
    env = make_env(args.env_name, discrete_action=True)
    num_state = env.observation_space[0].shape[0]
    num_action = env.action_space[0].n
    n_episodes = 100
    save_gifs = False
    episode_length = 150
    # nagents = args.adv_num
    nagents = args.agent_num
    ifi = 1 / 30
    gif_path = './gifs'
    success_rate = 0
    num_success = 0
    model_dir = args.model_dir
    #model_name = '5agents_120env_3000'
    # writer = SummaryWriter('/home/chenjy/pytorch-a2c-ppo-acktr-gail/logs'+model_dir)

    agents = []
    # for i in range(nagents):
    #     agents.append(MLPBase())
    # start = time.time()
    for i in range(nagents):
        # actor_critic, ob_rms = torch.load('/home/chenjy/new_version/trained_models/ppo' + args.model_dir + '/agent_%i' % (i+1) + ".pt")
        actor_critic, ob_rms = torch.load('/home/chenjy/new_version/trained_models/ppo' + args.model_dir + '/agent_1' + ".pt")
        agents.append(actor_critic)
    if not os.path.exists('./gifs/' + model_dir):
        os.makedirs('./gifs/' + model_dir)
    # end = time.time()
    # print('load', round(end-start,4))

    recurrent_hidden_states = torch.zeros(1,agents[0].recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    
    cover_rate_sum = 0

    starts = []
    starts_old = []
    hard_case = []
    buffer_length = 10000
    curri = 0
    N_new = 1000
    N_old = 500
    max_step = 0.1
    TB = 10
    M = 10000
    Rmin = 0.1
    Rmax = 0.5
    one_starts_landmark = []
    one_starts_agent = []
    for j in range(N_new):
        for i in range(nagents):
            landmark_location = np.random.uniform(-1, +1, 2) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        index_sample = BatchSampler(SubsetRandomSampler(range(nagents)),nagents,drop_last=True)
        for indices in index_sample:
            for k in indices:
                one_starts_agent.append(copy.deepcopy(one_starts_landmark[k]+0.01))
        # pdb.set_trace()
        starts.append(one_starts_agent+one_starts_landmark)
        starts_old.append(one_starts_agent+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []
    # hard case
    agent_location = np.zeros(2)
    for j in range(n_episodes):
        for i in range(nagents):
            agent_location = np.random.uniform(-1.0, -0.7, 2) 
            one_starts_agent.append(copy.deepcopy(agent_location))
        for i in range(int(nagents/4.0)):
            landmark_location = np.random.uniform(-0.5, +0.0, 2) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        for i in range(int(nagents/4.0),int(nagents/2.0)):
            landmark_location = np.random.uniform(+0.7, +1.0, 2) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        for i in range(int(nagents/2.0),int(3*nagents/4.0)):
            landmark_location[0] = np.random.uniform(-1.0, -0.7, 1) 
            landmark_location[1] = np.random.uniform(-0.5, +0.0, 1) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        for i in range(int(3*nagents/4.0),int(4*nagents/4.0)):
            landmark_location[0] = np.random.uniform(-0.5, +0.0, 1) 
            landmark_location[1] = np.random.uniform(-0.7, -1.0, 1) 
            one_starts_landmark.append(copy.deepcopy(landmark_location))
        hard_case.append(one_starts_agent+one_starts_landmark)
        one_starts_agent = []
        one_starts_landmark = []

    Rew = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for ep_i in range(n_episodes):
        print("Episode %i of %i" % (ep_i + 1, n_episodes))
        obs = env.reset(nagents)
        # obs = env.init_set(nagents,hard_case[ep_i])
        # sample_index = random.sample(range(len(starts)),1)

        # for i in range(nagents):
        #     x1.append(starts[sample_index[0]][i][0])
        #     y1.append(starts[sample_index[0]][i][1])
        #     x2.append(starts[sample_index[0]][i+nagents][0])
        #     y2.append(starts[sample_index[0]][i+nagents][1])
        # plt.scatter(x1,y1,c='#0000FF')
        # plt.scatter(x2,y2,c='#A52A2A')
        # x1 = []
        # y1 = []
        # x2 = []
        # y2 = []
        # if j % 1 == 0 :
        #     plt.savefig('/home/chenjy/new_version/test_init/test_'+str(ep_i)+'.jpg')
        #     plt.clf()
        # obs,rou_index= env.new_starts_obs(starts,nagents,0,sample_index)
        # pdb.set_trace()
        if save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        # env.render('human')
        for t_i in range(episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            state = torch.tensor([state for state in obs], dtype=torch.float).cuda()
            # get actions as torch Variables
            #import pdb; pdb.set_trace()
            ### 手动拼agent的state
            actions = []
            al_agent = []
            al_landmark = []
            share_obs = state.view(1,-1)
            # for i in range(nagents):
            #     share_obs.append(state[i].view(-1,num_state))
            # share_obs = np.array(share_obs).from_numpy()
            # start = time.time()
            for i in range(nagents):
                obs = state[i].view(-1, num_state)
                # value, action, _, recurrent_hidden_states, alpha_agent, alpha_landmark = agents[i].act(share_obs, obs, nagents, i, recurrent_hidden_states, masks)
                value, action, _, recurrent_hidden_states = agents[i].act(share_obs, obs, nagents, recurrent_hidden_states, masks, True)
                actions.append(action)
                # al_agent.append(alpha_agent)
                # al_landmark.append(alpha_landmark)
            torch_actions = actions
            end = time.time()
            # print('inference', round(end-start,4))
            #torch_actions = [agent1(state[0].view(-1, num_state)), agent2(state[1].view(-1, num_state)), agent3(state[2].view(-1, num_state)), agent4(state[3].view(-1, num_state)), agent5(state[4].view(-1, num_state))]

            # convert actions to numpy arrays
            prob_actions = [ac.data.cpu().numpy().flatten() for ac in torch_actions]
            #import pdb; pdb.set_trace()
            ### 手动生成one-hot向量
            actions = []
            for a in prob_actions:
                #index = np.argmax(a)
                ac = np.zeros(num_action)
                ac[a] = 1
                actions.append(ac)

            #import pdb; pdb.set_trace()
            # start = time.time()
            obs, rewards, dones, infos = env.step(actions)
            Rew.append(rewards)
            # print('step '+ str(t_i) + ': ' + str(rewards))
            #pdb.set_trace()
            # end = time.time()
            # print('env', round(end-start,4))
            masks.fill_(0.0 if dones else 1.0)
            #import pdb; pdb.set_trace()
            # start = time.time()
            if save_gifs:
                frames.append(env.render('rgb_array')[0])
            # end = time.time()
            # print('render one', round(end-start,4))
            # start = time.time()
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            # env.render('human')
            # end = time.time()
            # print('sleep', round(end-start,4))

            if t_i == episode_length -1:
                reach_num = num_reach(env.world)
                # writer.add_scalar('number_reach' , num_reach, ep_i)
                # writer.add_scalar('cover rate' , num_reach/nagents, ep_i)
                print('number_reach', reach_num)
                print('cover rate once', reach_num/nagents)
                cover_rate_sum = cover_rate_sum + reach_num/nagents
                # if reach_num/nagents < 1.0:
                #     print('obs: ', obs)
                # if reach_num/nagents == 1.0:
                #     starts = SampleNearby(starts, N_new, max_step, TB, M)
                # for a in env.world.agents:
                #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for l in env.world.landmarks]
                #     if min(dists) > env.world.agents[0].size + env.world.landmarks[0].size:
                #         print('agent_i: ', a.name)
                #         print('alpha_agent: ', al_agent[int(a.name[6:])])
                #         print('alpha_landmark: ', al_landmark[int(a.name[6:])])
                #         print('dists: ',dists)
                # for l in env.world.landmarks:
                #     dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in env.world.agents]
                #     if min(dists) > env.world.agents[0].size + env.world.landmarks[0].size:
                #         print('landmark_i: ', l.name)
        print('mean rewards: ', np.mean(Rew))
        Rew = []

        if save_gifs:
            gif_num = 0
            while os.path.exists('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i)):
                gif_num += 1
            imageio.mimsave('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i),
                            frames, duration=ifi)
    print('cover_rate', cover_rate_sum/n_episodes)
    # print('success rate',num_success/n_episodes)
