import numpy as np
import math
import  torch
from collections import defaultdict
import dill

class QLearning:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = config.learning_rate
        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.epsilon = 0
        self.sample_count = 0
        self.config = config
        self.Q_table = defaultdict(lambda: np.zeros(action_dim))

    def choose_action(self, state):
        self.sample_count += 1
        # epsilon decay
        self.epsilon = self.epsilon_end + (self.epsilon_start-self.epsilon_end)\
                       * math.exp(-1.0*self.sample_count/self.epsilon_decay)
        #epsilon-greedy
        if np.random.uniform(0,1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.action_dim)

        return action

    def predict(self , state):
        return np.argmax(self.Q_table[str(state)])

    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.learning_rate * (Q_target - Q_predict)

    def save(self, save_path):
        torch.save(
            obj=self.Q_table,
            f=save_path,
            pickle_module=dill
        )
        print("model successfully saved")

    def load(self, model_path):
        self.Q_table = torch.load(
            f=model_path
        )
        print("mode successfully loaded")

    def train(self, env):
        print("start training!")
        print(f'环境:{self.config.env_name}, 算法:{self.config.algo_name}, 设备:{self.config.device}')

        rewards = []
        ma_rewards = []  # 滑动平均奖励

        for eps in range(self.config.train_eps):
            eps_reward = 0
            state = env.reset()
            while True:
                action = self.choose_action(state)
                next_state , reward , done , _ = env.step(action)
                self.update(state,action,reward,next_state,done)
                state = next_state
                eps_reward += reward
                if done:
                    break

            rewards.append(eps_reward)
            #?
            if ma_rewards:
                ma_rewards.append(ma_rewards[-1] * 0.9 + eps_reward * 0.1)
            else:
                ma_rewards.append(eps_reward)

            print("eps:{}/{},reward:{:.1f},".format(eps+1,self.config.train_eps,eps_reward))
        print("training finfised")
        return rewards, ma_rewards

    def test(self, env):
        print("start testing!")
        print(f'环境:{self.config.env_name}, 算法:{self.config.algo_name}, 设备:{self.config.device}')

        for item in self.Q_table.items():
            print(item)

        rewards = []
        ma_rewards = []

        for eps in range(self.config.test_eps):
            eps_reward = 0
            state = env.reset()
            while True:
                action = self.predict(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                eps_reward += reward
                if done:
                    break

            rewards.append(eps_reward)
            # ?
            if ma_rewards:
                ma_rewards.append(ma_rewards[-1] * 0.9 + eps_reward * 0.1)
            else:
                ma_rewards.append(eps_reward)
            print("eps:{}/{},reward:{:.1f},".format(eps+1,self.config.train_eps,eps_reward))
        print("testing finfised")
        return rewards, ma_rewards