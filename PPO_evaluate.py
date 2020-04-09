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

# Parameters
gamma = 0.95
render = False
seed = 1
log_interval = 10

env = make_env("simple_spread", discrete_action=True)
num_state = env.observation_space[0].shape[0]
num_action = env.action_space[0].n
#torch.manual_seed(seed)
#env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False

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

if __name__ == '__main__':
    ### 手动加agent
    # agent1 = Actor()
    # agent2 = Actor()
    # agent3 = Actor()
    # agent4 = Actor()
    # agent5 = Actor()
    args = get_args()
    n_episodes = 100
    save_gifs = False
    episode_length = 100
    nagents = args.agent_num
    # nagents = args.agent_num
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
    for ep_i in range(n_episodes):
        print("Episode %i of %i" % (ep_i + 1, n_episodes))
        # obs = env.reset(nagents)
        obs = env.init_set(nagents)
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
                value, action, _, recurrent_hidden_states = agents[i].act(share_obs, obs, nagents, i, recurrent_hidden_states, masks)
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


        if save_gifs:
            gif_num = 0
            while os.path.exists('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i)):
                gif_num += 1
            imageio.mimsave('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i),
                            frames, duration=ifi)
    print('cover_rate', cover_rate_sum/n_episodes)
    # print('success rate',num_success/n_episodes)
