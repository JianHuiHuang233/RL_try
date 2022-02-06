import gym
import torch
from envs.gridworld_env import CliffWalkingWapper
from agent import QLearning
import sys
import os
from common.utils import make_dir,save_results,plot_rewards,plot_rewards_cn
import datetime

algo_name = 'Q-learning'  # 算法名称
env_name = 'CliffWalking-v0'  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 检测GPU

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 获取当前时间


class QlearningConfig:
    def __init__(self):
        self.algo_name = algo_name
        self.env_name = env_name
        self.device = device
        self.train_eps = 400
        self.test_eps = 50
        self.gamma = 0.9
        self.epsilon_start = 0.95
        self.epsilon_end = 0.1
        self.epsilon_decay = 300
        self.learning_rate = 0.1

class PlotConfig:
    ''' 绘图相关参数设置
    '''

    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = device # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片

def create_env_and_agent(cfg, seed=1):
    env = gym.make(cfg.env_name)
    env = CliffWalkingWapper(env)
    env.seed(seed)

    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    agent = QLearning(state_dim, action_dim, cfg)
    return env, agent


if __name__ == "__main__":
    cfg = QlearningConfig()
    plot_cfg = PlotConfig()

    #train
    env, agent = create_env_and_agent(cfg, seed=1)
    rewards, ma_rewards = agent.train(env)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)
    agent.save(plot_cfg.model_path + 'Qlearning_model.pkl')
    save_results(rewards,ma_rewards,tag="train",path=plot_cfg.result_path)
    plot_rewards(rewards,ma_rewards,plot_cfg,tag="train")

    #test
    env, agent = create_env_and_agent(cfg,seed=10)
    agent.load(plot_cfg.model_path + 'Qlearning_model.pkl')
    rewards, ma_rewards = agent.test(env)
    save_results(rewards, ma_rewards, tag="test", path=plot_cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="test")




