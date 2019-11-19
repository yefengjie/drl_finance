import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""
    """演员（策略）模型"""

    def __init__(self, state_size, action_size, seed, fc1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        初始化参数并构建模型
        Params
        ======
            state_size (int): Dimension of each state 每个状态的维度
            action_size (int): Dimension of each action 每个动作的维度
            seed (int): Random seed 随机种子
            fc1_units (int): Number of nodes in first hidden layer 第一个隐藏层中的节点数
            fc2_units (int): Number of nodes in second hidden layer 第二个隐藏层中的节点数
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        """构建一个演员(策略)网络以映射 状态->动作"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""
    """批评家(值)模型"""

    def __init__(self, state_size, action_size, seed, fcs1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        初始化参数并构建模型
        Params
        ======
            state_size (int): Dimension of each state 每个状态的维度
            action_size (int): Dimension of each action 每个动作的维度
            seed (int): Random seed 随机种子
            fcs1_units (int): Number of nodes in the first hidden layer 第一个隐藏层中的节点数
            fc2_units (int): Number of nodes in the second hidden layer 第二个隐藏层中的节点数
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        """构建一个批评家(值)网络以映射(状态，动作)对-> Q值"""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
