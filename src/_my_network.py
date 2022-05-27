import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')


class Tabular_Q():
    def __init__(self, shape):
        r_min,r_max=1,1
        self.eval=np.random.uniform(r_min, r_max, shape)
        self.tar=np.random.uniform(r_min, r_max, shape)


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, NUM_STATES, NUM_ACTIONS):
        super(Net, self).__init__()
        self.layer=6
        self.units=64
        self.input = nn.Linear(NUM_STATES, self.units).to(device)
        self.input.weight.data.normal_(0,0.1).to(device)
        self.mid = []
        for i in range(self.layer-2):
            self.mid.append(nn.Linear(self.units,self.units).to(device))
            self.mid[i].weight.data.normal_(0,0.1).to(device)
        self.out = nn.Linear(self.units,NUM_ACTIONS).to(device)
        self.out.weight.data.normal_(0,0.1).to(device)

    def forward(self, x):
        x = x.to(device)
        x = self.input(x)
        x = F.relu(x)
        for i in range(self.layer-2):
            x = self.mid[i](x)
            x = F.relu(x)
        x = self.out(x)
        return x


class DQN():  # Deep Q network model for a single RM
    """docstring for DQN"""
    def __init__(self, state_num, action_num):
        self.eval_net = Net(state_num, action_num).to(device)
        self.target_net = Net(state_num, action_num).to(device)
        # self.target_net[i].load_state_dict(torch.load('net_param'+str(i)))





