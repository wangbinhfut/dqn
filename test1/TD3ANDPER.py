# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import random
import torch.optim as optim
import csv


device = torch.device("cpu")
import matplotlib.pyplot as plt

# params = {
#     'gamma': 0.98,
#     'actor_lr': 0.0001,
#     'critic_lr': 0.01,
#     'tau': 0.001,
#     'capacity': 200,
#     'batch_size': 32,
#     'policy_noise': 0.2,
#     'noise_clip': 0.5,
#     'policy_freq': 2,
# }  ###############

class SumTree:
    def __init__(self, capacity: int):
        # 初始化SumTree，设定容量
        self.capacity = capacity
        # 数据指针，指示下一个要存储数据的位置
        self.data_pointer = 0
        # 数据条目数
        self.n_entries = 0
        # 构建SumTree数组，长度为(2 * capacity - 1)，用于存储树结构
        self.tree = np.zeros(2 * capacity - 1)
        # 数据数组，用于存储实际数据
        self.data = np.zeros(capacity, dtype=object)

    def update(self, tree_idx, p):#更新采样权重
        # 计算权重变化
        change = p - self.tree[tree_idx]
        # 更新树中对应索引的权重
        self.tree[tree_idx] = p

        # 从更新的节点开始向上更新，直到根节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):#向SumTree中添加新数据
        # 计算数据存储在树中的索引
        tree_idx = self.data_pointer + self.capacity - 1
        # 存储数据到数据数组中
        self.data[self.data_pointer] = data
        # 更新对应索引的树节点权重
        self.update(tree_idx, p)

        # 移动数据指针，循环使用存储空间
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        # 维护数据条目数
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):#采样数据
        # 从根节点开始向下搜索，直到找到叶子节点
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # 如果左子节点超出范围，则当前节点为叶子节点
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 根据采样值确定向左还是向右子节点移动
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        # 计算叶子节点在数据数组中的索引
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])


class ReplayTree:  # ReplayTree for the per(Prioritized Experience Replay) DQN.
    def __init__(self, capacity):
        self.capacity = capacity  # 记忆回放的容量
        self.tree = SumTree(capacity)  # 创建一个SumTree实例
        self.abs_err_upper = 10.  # 绝对误差上限
        self.epsilon = 0.01
        ## 用于计算重要性采样权重的超参数
        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6  # 决定优先级的使用程度，α = 0 对应均匀采样
        self.beta = 0.4

    def __len__(self):  # 返回存储的样本数量
        return self.tree.total()

    def push(self, error, sample):  # Push the sample into the replay according to the importance sampling weight
        p = (np.abs(error.detach().numpy()) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, batch_size):
        pri_segment = self.tree.total() / batch_size  #priority segment，优先级分割，因为只需要32个数据，均分32组，每组选一个数据

        priorities = []
        batch = []
        idxs = []

        is_weights = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            prob = p / self.tree.total()

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return zip(*batch), idxs, is_weights

    def batch_update(self, tree_idx, abs_errors):  # Update the importance sampling weight
        abs_errors += self.epsilon

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        a = torch.sigmoid(self.linear3(x))

        return a


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Q1 architecture
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.normal_(0, 0.1)

        # Q2 architecture
        self.linear4 = nn.Linear(input_size, hidden_size)
        self.linear4.weight.data.normal_(0, 0.3)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        self.linear5.weight.data.normal_(0, 0.3)
        self.linear6 = nn.Linear(hidden_size, output_size)
        self.linear6.weight.data.normal_(0, 0.3)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)

        x2 = F.relu(self.linear4(x))
        x2 = F.relu(self.linear5(x2))
        q2 = self.linear6(x2)
        return q1, q2

    def Q1(self, s, a):
        x = torch.cat([s, a], 1)
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)
        return q1

    def Q2(self, s, a):  #用于存储历史数据时计算Q值误差
        x = torch.cat([s, a], 0)
        x1 = F.relu(self.linear1(x))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)
        return q1


class TD3Agent(object):
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

            s_dim = 4
            a_dim = 1
            TD3MEMORY_CAPACITY = 1000

            self.actor = Actor(s_dim, 32, a_dim).to(device)
            self.actor_target = Actor(s_dim, 32, a_dim).to(device)
            self.critic = Critic(s_dim + a_dim, 32, 1).to(device)
            self.critic_target = Critic(s_dim + a_dim, 32, 1).to(device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
            self.TD3memory_counter = 0
            self.TD3memory=ReplayTree(capacity=TD3MEMORY_CAPACITY)
            self.learn_start = 1000
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.total_it = 0  #总学习次数

        def act(self, s0):

            s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0).to(device)
            a0 = self.actor(s0).squeeze(0).detach().cpu().numpy()
            # print(s0, a0)
            return a0

        def put(self, s, a, r, s1, done):
            next_action = self.actor_target(torch.tensor(s1, dtype=torch.float)).detach()
            target_Q = self.critic_target.Q2(torch.tensor(s1, dtype=torch.float), torch.tensor(next_action, dtype=torch.float)).detach()
            r = torch.tensor(r, dtype=torch.float)
            done = torch.tensor(done, dtype=torch.float)
            target_Q = r + done * self.gamma * target_Q
            current_Q= self.critic.Q2(torch.tensor(s, dtype=torch.float), torch.tensor(a, dtype=torch.float)).detach()
            transition = (s, a, r, s1, done)
            error = abs(current_Q -target_Q)
            self.TD3memory.push(error, transition)  # 添加经验和初始优先级
            self.TD3memory_counter += 1


        def clear_buffer(self, step):
            if (step + 33) % 10000 == 0:
                self.buffer = []

        def learn(self):
            self.total_it += 1
            if self.TD3memory_counter < self.learn_start:
                return
            batch, tree_idx, is_weights = self.TD3memory.sample(self.batch_size)
            s0, a0, r1, s1, done = batch
            s0 = torch.tensor(s0, dtype=torch.float).to(device)
            a0 = torch.tensor(a0, dtype=torch.float).to(device)
            r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1).to(device)
            s1 = torch.tensor(s1, dtype=torch.float).to(device)
            done = torch.tensor(done, dtype=torch.float).to(device).unsqueeze(1)   #.unsqueeze(1) 要将形状为 (32,) 的张量转换为形状为 (32, 1) 的张量
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(a0) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

                next_action = ( self.actor_target(s1) + noise).clamp(0, 1)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(s1, next_action)
                target_Q_aa = torch.min(target_Q1, target_Q2)
                target_Qa1 = self.gamma * target_Q_aa
                target_Qa2 = done * target_Qa1
                target_Q = r1 + target_Qa2
                #target_Q = r1 + (1 - done) * self.gamma * target_Q_aa
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(s0, a0)

            # Compute critic loss
            critic_loss = (torch.FloatTensor(is_weights) * (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))).mean()
            critic_losslist= critic_loss.tolist()
            with open('./downloss.csv', mode='a', newline='', encoding='utf-8') as file:
                fw = csv.writer(file)
                fw.writerow([critic_losslist])
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            abs_errors = torch.abs(current_Q1 - target_Q).detach().numpy().squeeze()
            self.TD3memory.batch_update(tree_idx, abs_errors)  # 更新经验的优先级
            # Delayed policy updates
            
            if self.total_it % self.policy_freq == 0:

                # Compute actor losse
                actor_loss = -self.critic.Q1(s0, self.actor(s0)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


