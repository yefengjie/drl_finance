import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size 经验回放缓存区大小
BATCH_SIZE = 128        # minibatch size 小批量大小
GAMMA = 0.99            # discount factor 折扣系数
TAU = 1e-3              # for soft update of target parameters 用于目标参数的软更新
LR_ACTOR = 1e-4         # learning rate of the actor 演员学习率
LR_CRITIC = 1e-3        # learning rate of the critic 批评家学习率
WEIGHT_DECAY = 0        # L2 weight decay L2重量衰减

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    """与环境互动并从中学习"""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        “”“ 初始化一个代理对象 ”“”
        
        Params
        ======
            state_size (int): dimension of each state 每个状态的维度
            action_size (int): dimension of each action 每个动作的维度
            random_seed (int): random seed 随机种子
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        # 演员网络(w/目标网络)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        # 批判家网络(w/目标网络)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        # 噪声过程
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        # 经验回放内存
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        """将经验保存在回放内存中，并使用缓冲区中的随机样本进行学习"""
        # Save experience / reward
        # 保存经验/奖励
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        # 如果样本足够了，开始学习
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        """根据当前策略指定的状态返回动作"""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        action = (action + 1.0) / 2.0
        return np.clip(action, 0, 1)


    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        使用给定的批量经验元组更新策略和值参数
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor 折扣系数
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # ---------------------------- 更新批评家 ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # 从目标模型预测下一状态动作和Q值
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        # 计算当前状态(y_i)的Q目标
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        # 计算批判家的损失
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        # 最小化损失
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # ---------------------------- 更新演员 ---------------------------- #
        # Compute actor loss
        # 计算演员的损失
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        # 最小化损失
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        # ----------------------- 更新目标网络 ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        软更新模型参数
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 插值参数
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""
    """Ornstein-Uhlenbeck 过程."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        """初始化参数和噪声处理"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        """将内部状态(=noise)重置为均值(mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        """更新内部状态并将其作为噪声样本返回"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    """固定大小的缓冲区，用于存储经验元组"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        初始化ReplayBuffer对象
        Params
        ======
            buffer_size (int): maximum size of buffer 缓存区的最大大小
            batch_size (int): size of each training batch 每个训练批次大小
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque) 内部存储器(双端队列 )
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        """将新的经验添加到内存中"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        """从内存中随机取样一批经验值"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        """返回内部存储器的当前大小"""
        return len(self.memory)