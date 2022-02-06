
def train(cfg, env, agent):
    print("start training!")
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')

    rewards = []
    ma_rewards = [] #滑动平均奖励

    for eps in range(cfg.train_eps):
        eps_reward = 0
        state = env.reset()
        while True:
            action = agent.choose_action(state)