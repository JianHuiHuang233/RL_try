import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.cur = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.cur] = (state, action, reward, next_state, done)
        self.cur = (self.cur + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        # print(state, action, reward, next_state, done)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DoubleDQN:
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # print(state_dim , action_dim)
        self.gamma = cfg.gamma
        self.frame_idx = 0
        self.device = cfg.device
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        for target_para, para in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_para.data.copy_(para.data)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.cfg = cfg

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                # print("state:{}".format(state))
                q_vals = self.policy_net(state)
                # choose the action with max_value
                action = q_vals.max(1)[1].item()
        else:
            action = random.randrange(self.action_dim)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        # print(reward_batch)
        state_batch = torch.tensor(state_batch,device=self.device,dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float)

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        #for DDQN  ========================
        next_q_values = self.policy_net(next_state_batch)
        next_target_values = self.target_net(next_state_batch)
        next_target_q_values = next_target_values.gather(1, torch.max(next_q_values,1)[1].unsqueeze(1)).squeeze(1)
        expect_q_values = reward_batch + self.gamma * next_target_q_values * (1-done_batch)
        #==================================

        # print(q_values.shape, expect_q_values.unsqueeze(1).shape)
        loss = nn.MSELoss()(q_values , expect_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        for para in self.policy_net.parameters():
            para.grad.data.clamp_(-1,1)

        self.optimizer.step()


    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_para, para in zip(self.target_net.parameters(), self.policy_net.parameters()):
            para.data.copy_(target_para.data)

    def train(self, env):
        print("start train")
        print(f'环境：{self.cfg.env_name}, 算法：{self.cfg.algo_name}, 设备：{self.cfg.device}')

        rewards = []
        ma_rewards = []

        for eps in range(self.cfg.train_eps):
            eps_reward = 0
            state = env.reset()
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.memory.push(state,action,reward,next_state,done)
                state = next_state
                self.update()

                eps_reward += reward
                if done:
                    break

            if (eps+1) % self.cfg.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            rewards.append(eps_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * eps_reward)
            else:
                ma_rewards.append(eps_reward)

            if (eps + 1) % 10 == 0:
                print('回合：{}/{}, 奖励：{}'.format(eps + 1, self.cfg.train_eps, eps_reward))

        print("train done!")
        env.close()
        return rewards, ma_rewards

    def test(self, env):
        print("start test")
        print(f'环境：{self.cfg.env_name}, 算法：{self.cfg.algo_name}, 设备：{self.cfg.device}')

        ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
        self.cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
        self.cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
        ################################################################################

        rewards = []
        ma_rewards = []

        for eps in range(self.cfg.test_eps):
            eps_reward = 0
            state = env.reset()
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                eps_reward += reward
                if done:
                    break

            rewards.append(eps_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * eps_reward)
            else:
                ma_rewards.append(eps_reward)

            if (eps + 1) % 10 == 0:
                print('回合：{}/{}, 奖励：{}'.format(eps + 1, self.cfg.train_eps, eps_reward))

        print("test done!")
        env.close()
        return rewards, ma_rewards


