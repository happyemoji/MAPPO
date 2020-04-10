import torch.nn as nn
import torch.nn.functional as F
import torch

class ATTNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ATTNetwork, self).__init__()
        self.input_dim = input_dim
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)   #first embedding layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  #agent i fc layer
        self.fc3 = nn.Linear(hidden_dim*2, out_dim)     #last fc layer
        # self.nonlin = nonlin
        # if constrain_out and not discrete_action:
        #     # initialize small to prevent saturation
        #     self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        #     self.out_fn = F.tanh
        # else:  # logits for discrete action (will softmax later)
        #     self.out_fn = lambda x: x
        
        #correlation matrix
        self.correlation_mat = nn.Parameter(torch.FloatTensor(hidden_dim,hidden_dim),requires_grad=True)
        self.correlation_mat.data.fill_(0.25)
        #self.correlation_mat2 = nn.Parameter(torch.FloatTensor(hidden_dim,hidden_dim),requires_grad=True)
        #self.correlation_mat2.data.fill_(0.25)

    # def forward(self, X):
    #     """
    #     Inputs:
    #         X (PyTorch Matrix): Batch of observations
    #     Outputs:
    #         out (PyTorch Matrix): Output of network (actions, values, etc)
    #     """
    #     h1 = self.nonlin(self.fc1(self.in_fn(X)))
    #     h2 = self.nonlin(self.fc2(h1))
    #     out = self.out_fn(self.fc3(h2))
    #     return out

    def forward(self, inputs, n_agents, agent_i):
        batch_size = int(inputs.size(0)/n_agents)
        inputs = self.in_fn(inputs)
        fi = F.relu(self.fc1(inputs))
        fi = fi.view(batch_size,n_agents,-1)   #(batch_size,n_agents,embedding_dim)
        beta = []
        f_j = []
        for j in range(n_agents):
            if j!=agent_i:
                f_j.append(fi[:,j].view(batch_size,1,-1))    #(batch_size,1,eb_dim)
                beta_i_j = torch.matmul(fi[:,agent_i].view(batch_size,1,-1),self.correlation_mat)
                #beta_i_j = torch.matmul(beta_i_j,self.correlation_mat2)
                beta_i_j = torch.matmul(beta_i_j,fi[:,j].view(batch_size,-1,1))
                beta.append(beta_i_j.squeeze(1).squeeze(1))
        f_j = torch.stack(f_j,dim = 1).squeeze(2)  #(batch_size,n_agents-1,eb_dim)
        beta = torch.stack(beta,dim = 1)            
        alpha = F.softmax(beta,dim = 1).unsqueeze(2)  #(batch_size,n_agents-1,1)
        vi = torch.mul(alpha,f_j)
        vi = torch.sum(vi,dim = 1).unsqueeze(1) #(batch_size,1,eb_dim)
        fc2_outputs = F.relu(self.fc2(fi[:,agent_i]))
        fc3_inputs = torch.cat([fc2_outputs.view(batch_size,1,-1),vi],dim=2)
        q_i = self.fc3(fc3_inputs)
        return q_i.squeeze(1)
