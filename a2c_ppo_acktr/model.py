import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
import time

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, agent_num, agent_i, base=None, dist=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
                self.base = base(obs_shape[0], agent_num, **base_kwargs)
            elif len(obs_shape) == 1:
                base = MLPBase
                #base = ATTBase
                self.base = base(obs_shape[0], agent_num, **base_kwargs)
            else:
                raise NotImplementedError
        else:
            self.base = base
        # self.base = base(obs_shape[0], **base_kwargs)
        # actor输入维度num_state，critic输入num_state*agent_num

        # self.base = base(obs_shape[0], agent_num, agent_i, **base_kwargs)

        #import pdb; pdb.set_trace()
        self.agent_i = agent_i

        if dist is None:
            if action_space.__class__.__name__ == "Discrete":
                num_outputs = action_space.n
                self.dist = Categorical(self.base.output_size, num_outputs)
            elif action_space.__class__.__name__ == "Box":
                num_outputs = action_space.shape[0]
                self.dist = DiagGaussian(self.base.output_size, num_outputs)
            elif action_space.__class__.__name__ == "MultiBinary":
                num_outputs = action_space.shape[0]
                self.dist = Bernoulli(self.base.output_size, num_outputs)
            else:
                raise NotImplementedError
        else:
            self.dist = dist

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, share_inputs, inputs, agent_num, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(share_inputs, inputs, agent_num, self.agent_i, rnn_hxs, masks)
        #value, actor_features, rnn_hxs, alpha_agent, alpha_landmark = self.base(share_inputs, inputs, self.agent_i, rnn_hxs, masks)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, share_inputs, inputs, agent_num, rnn_hxs, masks):
        value, _, _ = self.base(share_inputs, inputs, agent_num, self.agent_i, rnn_hxs, masks)
        return value

    def evaluate_actions(self, share_inputs, inputs, agent_num, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(share_inputs, inputs, agent_num, self.agent_i, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, agent_num, recurrent=False, assign_id=False, hidden_size=100):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        if assign_id:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs + agent_num, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs * agent_num + agent_num, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        else:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs * agent_num, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, share_inputs, inputs, agent_i, rnn_hxs, masks):
        #import pdb; pdb.set_trace()
        share_obs = share_inputs
        obs = inputs

        #if self.is_recurrent:
        #    x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(share_inputs)
        hidden_actor = self.actor(inputs)
        
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class ObsEncoder(nn.Module):
    def __init__(self, hidden_size=100):
        super(ObsEncoder, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        '''                        
        self.self_encoder = nn.Linear(4, hidden_size)
        self.other_agent_encoder = nn.Linear(2, hidden_size)
        self.landmark_encoder = nn.Linear(2, hidden_size)
        self.agent_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        self.agent_correlation_mat.data.fill_(0.25)
        self.landmark_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        self.landmark_correlation_mat.data.fill_(0.25)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.encoder_linear = nn.Linear(3*hidden_size, hidden_size)
        '''
        self.self_encoder = nn.Sequential(
                            init_(nn.Linear(4, hidden_size)), nn.Tanh())
        self.other_agent_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh())
        self.landmark_encoder = nn.Sequential(
                            init_(nn.Linear(2, hidden_size)), nn.Tanh())
        self.agent_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.agent_correlation_mat.data, gain=1)
        self.landmark_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.landmark_correlation_mat.data, gain=1)
        self.fc = nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        #self.encoder_linear = init_(nn.Linear(3*hidden_size, hidden_size))
        # 加上激活函数 效果会有比较大的提升 虽然还是达不到标准
        self.encoder_linear = nn.Sequential(
                            init_(nn.Linear(hidden_size * 3, hidden_size)), nn.Tanh(),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

    # agent_num需要手动设置一下
    def forward(self, inputs, agent_num):
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        landmark_num = int((obs_dim-4)/2)-2*(agent_num-1)
        #landmark_num = int((obs_dim-4-4*(agent_num-1))/3)
        #import pdb; pdb.set_trace()
        self_emb = self.self_encoder(inputs[:, :4])
        other_agent_emb = []
        beta_agent = []
        landmark_emb = []
        beta_landmark = []
        #start = time.time()

        agent_beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.agent_correlation_mat)
        landmark_beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.landmark_correlation_mat) 

        for i in range(agent_num - 1):
            other_agent_emb.append(inputs[:, 4+2*landmark_num+2*i:4+2*landmark_num+2*(i+1)])
            #one_agent_emb = self.other_agent_encoder(inputs[:, 4+2*landmark_num+2*i:4+2*landmark_num+2*(i+1)])
            #import pdb; pdb.set_trace()
            #other_agent_emb.append(one_agent_emb)
            #beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.agent_correlation_mat)                #[batch_size, 1, hidden_size]
            #beta_ij = torch.matmul(agent_beta_ij, one_agent_emb.view(batch_size,-1,1))       #[batch_size, 1, 1]
            #beta_agent.append(beta_ij.squeeze(1).squeeze(1))
        for i in range(landmark_num):
            landmark_emb.append(inputs[:, 4+2*i:4+2*(i+1)])
            #one_landmark_emb = self.landmark_encoder(inputs[:, 4+2*i:4+2*(i+1)])
            #landmark_emb.append(one_landmark_emb)
            #beta_ij = torch.matmul(self_emb.view(batch_size,1,-1), self.landmark_correlation_mat)                #[batch_size, 1, hidden_size]
            #beta_ij = torch.matmul(landmark_beta_ij, one_landmark_emb.view(batch_size,-1,1))       #[batch_size, 1, 1]
            #beta_landmark.append(beta_ij.squeeze(1).squeeze(1))
        other_agent_emb = torch.stack(other_agent_emb,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        other_agent_emb = self.other_agent_encoder(other_agent_emb)
        #beta_agent = torch.stack(beta_agent,dim = 1) 
        beta_agent = torch.matmul(agent_beta_ij, other_agent_emb.permute(0,2,1)).squeeze(1)
        landmark_emb = torch.stack(landmark_emb,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        landmark_emb = self.landmark_encoder(landmark_emb)
        #beta_landmark = torch.stack(beta_landmark,dim = 1) 
        beta_landmark = torch.matmul(landmark_beta_ij, landmark_emb.permute(0,2,1)).squeeze(1)
        alpha_agent = F.softmax(beta_agent,dim = 1).unsqueeze(2)   
        alpha_landmark = F.softmax(beta_landmark,dim = 1).unsqueeze(2)
        other_agent_vi = torch.mul(alpha_agent,other_agent_emb)
        # other_agent_vi = self.layer_norm_1(other_agent_vi)
        other_agent_vi = torch.sum(other_agent_vi,dim=1)
        landmark_vi = torch.mul(alpha_landmark,landmark_emb)
        # landmark_vi = self.layer_norm_2(landmark_vi)
        landmark_vi = torch.sum(landmark_vi,dim=1)
        #end = time.time()
        #print("time: ", end-start)
        #import pdb; pdb.set_trace()
        gi = self.fc(self_emb)
        f = self.encoder_linear(torch.cat([gi, other_agent_vi, landmark_vi], dim=1))
        return f
        



class ATTBase(NNBase):
    def __init__(self, num_inputs, agent_num, recurrent=False, assign_id=False, hidden_size=100):
        super(ATTBase, self).__init__(recurrent, num_inputs, hidden_size)
        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.agent_num = agent_num
        #是否要用attention-based policy network    
        #self.actor = nn.Sequential(
        #        init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
        #        init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
    
        self.actor = ObsEncoder(hidden_size=hidden_size)
        #self.encoder = init_(nn.Linear(num_inputs, hidden_size))
        self.encoder = ObsEncoder(hidden_size=hidden_size)

        self.correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        #self.correlation_mat.data.fill_(0.25)
        nn.init.orthogonal_(self.correlation_mat.data, gain=1)

        #self.fc = init_(nn.Linear(hidden_size, hidden_size))
        self.fc = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = nn.Sequential(
                init_(nn.Linear(hidden_size * 2, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, 1)))

        self.layer_norm_1 = nn.LayerNorm(hidden_size)

        self.train()

    def forward(self, share_inputs, inputs, agent_num, agent_i, rnn_hxs, masks):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        #start = time.time()
        hidden_actor = self.actor(inputs, agent_num)
        #hidden_actor, alpha_agent, alpha_landmark = self.actor(inputs)
        #f_ii = self.encoder(inputs)    #[batch_size, hidden_size]
        #f_ii = self.encoder(inputs, self.agent_num)
        
        f_ii = self.encoder(inputs, agent_num)
        obs_beta_ij = torch.matmul(f_ii.view(batch_size,1,-1), self.correlation_mat)
        obs_encoder = []
        beta = []
        for i in range(agent_num):
        #for i in range(6):
            if i != agent_i:
                #obs_encoder.append(share_inputs[:, i*obs_dim:(i+1)*obs_dim])
                f_ij = self.encoder(share_inputs[:, i*obs_dim:(i+1)*obs_dim], agent_num)     #[batch_size, hidden_size]
                obs_encoder.append(f_ij)
                #beta_ij = torch.matmul(f_ii.view(batch_size,1,-1), self.correlation_mat)                #[batch_size, 1, hidden_size]
                #beta_ij = torch.matmul(obs_beta_ij, f_ij.view(batch_size,-1,1))       #[batch_size, 1, 1]
                #obs_encoder.append(f_ij)
                #beta.append(beta_ij.squeeze(1).squeeze(1))
        obs_encoder = torch.stack(obs_encoder,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        #beta = torch.stack(beta,dim = 1)  
        beta = torch.matmul(obs_beta_ij, obs_encoder.permute(0,2,1)).squeeze(1)
        alpha = F.softmax(beta,dim = 1).unsqueeze(2)
        vi = torch.mul(alpha,obs_encoder)
        # vi = self.layer_norm_1(vi)
        vi = torch.sum(vi,dim = 1)
        gi = self.fc(f_ii)
        value = self.critic_linear(torch.cat([gi, vi], dim=1))
        #end = time.time()
        #print("ATTBase time: ", end-start)
        #import pdb; pdb.set_trace()
        
        #value = 0
        return value, hidden_actor, rnn_hxs

    