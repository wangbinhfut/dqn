import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import csv
BATCH_SIZE=200
LR=0.001
EPSILON_START = 0.9  # epsilon 的初始值
EPSILON_END = 0.1  # epsilon 的最终值
EPSILON_DECAY = 1000 # epsilon 的衰减步数
GAMMA=0.5
TARGET_REPLACE_ITER= 10
MEMORY_CAPACITY= 1024
N_ACTIONS= 3
N_STATES= 5
n_mid = 64

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


class DQNNet(torch.nn.Module):
    def __init__(self,N_STATES,n_mid,N_ACTIONS):
        super(DQNNet,self).__init__()
        self.fc1 = torch.nn.Linear(N_STATES,n_mid)
        self.fc1.weight.data.normal_(0.1, 0.4)
        self.fc2 = torch.nn.Linear(n_mid,n_mid)
        self.fc2.weight.data.normal_(0.1, 0.4)
        self.fc3_adv = torch.nn.Linear(n_mid,N_ACTIONS)
        self.fc3_adv.weight.data.normal_(0.1, 0.4)
        self.fc3_v = torch.nn.Linear(n_mid,1)
        self.fc3_v.weight.data.normal_(0.1, 0.4)
    def forward(self,x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        adv = self.fc3_adv(h2)
        # expand函数作用是采用复制的方式扩展一个维度的长度，-1指不改变长度，这里把V复制成和A数量一样多
        val = self.fc3_v(h2)
        # V + A ,然后都减去A的平均值
        #output = val + adv - adv.mean(1,keepdim = True)
        output = val + adv - adv.mean()
        return output
    def forward1(self,x):
        h1 = F.relu(self.fc1(x))
        output = F.relu(self.fc2(h1))
        return output


class DQN(object):
    def __init__(self):
        self.eval_net = DQNNet(N_STATES,n_mid, N_ACTIONS)
        self.target_net1 = DQNNet(N_STATES,n_mid, N_ACTIONS)
        self.target_net2 = DQNNet(N_STATES, n_mid, N_ACTIONS)
        self.learn_step_counter=0
        self.memory_counter=0
        self.epsilon = EPSILON_START
        self.memory=ReplayTree(capacity=MEMORY_CAPACITY)
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()


        
    # 输出中间特征矩阵
    def get_eigenmatrix(self, state):
        state = torch.FloatTensor(state)
        eigenmatrix = self.eval_net.forward1(state)
        return eigenmatrix
    # ξ贪婪探索动作(2)     
    # 输出Q值
    def get_Qvalue(self, state):
        state = torch.FloatTensor(state)
        q_value = self.eval_net.forward(state).detach()
        return q_value
    # ξ贪婪探索动作(2)
    def get_action(self, state, action_set):
        if random.random() <= self.epsilon:  # 从可选的动作集中随机选择
            return random.choice(action_set)
        else:  # 在满足约束条件的动作集中选择代价最小的Q值
            state = torch.FloatTensor(state)
            q_value = self.eval_net.forward(state).detach()
            action_set = torch.LongTensor(action_set)
            action_index = int(q_value.gather(0, action_set).max(0)[1])  # 注意这里输出的action_choose是新的q_value的下标，还需要再对应到action_set
            action_choose = action_set[action_index]
            return action_choose

    # 完全贪婪进行评估
    def get_estimate_action(self, state, action_set):
        state = torch.FloatTensor(state)
        q_value = self.eval_net.forward(state).detach()
        action_set = torch.LongTensor(action_set)
        action_index = int(q_value.gather(0, action_set).max(0)[1])  # 注意这里输出的action_choose是新的q_value的下标，还需要再对应到action_set
        action_choose = action_set[action_index]
        return action_choose

    def store_transition(self,s,a,r,s_):
        policy_val =self.eval_net(torch.tensor(s, dtype=torch.float))[a]
        target_val1=self.target_net1(torch.tensor(s_, dtype=torch.float))
        target_val2 = self.target_net2(torch.tensor(s_, dtype=torch.float))
        transition = (s, a, r, s_)
        error = abs(policy_val-r-GAMMA*torch.min(torch.max(target_val1), torch.max(target_val2)))
        self.memory.push(error, transition)  # 添加经验和初始优先级
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 同步两个网络
            # self.target_net.load_state_dict(self.eval_net.state_dict())
            for target_param, param in zip(self.target_net1.parameters(), self.eval_net.parameters()):   #软更新
                target_param.data.copy_(target_param.data * 0.4 + param.data * 0.6)
            for target_param, param in zip(self.target_net2.parameters(), self.eval_net.parameters()):   #软更新
                target_param.data.copy_(target_param.data * 0.4 + param.data * 0.6)
        self.learn_step_counter += 1

        batch, tree_idx, is_weights = self.memory.sample(BATCH_SIZE)
        b_s,b_a,b_r,b_s_= batch
        b_s = torch.FloatTensor(b_s)
        b_a = torch.unsqueeze(torch.LongTensor(b_a), 1)
        b_r = torch.unsqueeze(torch.FloatTensor(b_r), 1)
        b_s_ = torch.FloatTensor(b_s_)
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next1 = self.target_net1(b_s_)
        q_next2 = self.target_net2(b_s_)
        q_target = b_r + GAMMA * torch.min( q_next1.max(1)[0].view(BATCH_SIZE, 1),q_next2.max(1)[0].view(BATCH_SIZE, 1))
        loss = (torch.FloatTensor(is_weights) * self.loss_func(q_eval, q_target)).mean()
        losslist= loss.tolist()
        with open('./uploss.csv', mode='a', newline='', encoding='utf-8') as file:
            fw = csv.writer(file)
            fw.writerow([losslist])
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()        
        abs_errors = torch.abs(q_eval - q_target).detach().numpy().squeeze()
        self.memory.batch_update(tree_idx, abs_errors)  # 更新经验的优先级
        # 更新 epsilon
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * self.learn_step_counter / EPSILON_DECAY) if EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1.0 * self.learn_step_counter / EPSILON_DECAY) > EPSILON_END \
            else EPSILON_END
        self.learn_step_counter += 1
