"""
@File: main_t_alltime.py
@author: Chenglong Xiao
@Describtion: 
"""
from matplotlib import pyplot as plt
import pandas as pd
from env_ddpg import ENV_DDPG
from env_ddpg_local import ENV_DDPG_LOCAL
from env_ddpg_nogan import ENV_DDPG_NOGAN
from env_ddpg_nocache import ENV_DDPG_NOCACHE
from network import Agent
import numpy as np

def ddpg(episode, step, num):
    # tcar为需卸载车辆， scar为服务车辆
    num_car = 21
    num_scar = int(num_car * 1 / 3)
    num_tcar = num_car - num_scar

    # 固定任务数量为10
    num_task = 10

    env = ENV_DDPG(num_car, num_tcar, num_scar, num_task)
    n_actions = 2
    n_state = 48
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    score_record = []
    score_record_step = []
    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    time_record = []
    time_record_step = []
    print('服务车辆数量： {}'.format(num_scar))

    for i in range(episode):
        score = 0
        obs = env.get_init_state()
        done = False

        while not done:
            act = MECSnet.choose_action(obs)
            new_state, reward, done = env.step(act)
            MECSnet.remember(obs, act, reward, new_state, int(done))
            MECSnet.learn()
            score += reward
            obs = new_state

        episode_record.append(i)
        cost_record.append(-score)
        time_record.append(env.t_all)
        # print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            # cost_record_step.append(np.mean(cost_record))
            time_record_step.append(np.mean(time_record))

    df = pd.DataFrame({"Episode": episode_record_step, "Time": time_record_step}).set_index('Episode')
    df.to_excel("T_alltime/episode_alltime_ddpg/episode_alltime_ddpg_{}.xlsx".format(num))

    plt.figure()
    x_data = range(len(time_record_step))
    plt.plot(x_data, time_record_step)

    plt.show()

def ddpg_local(episode, step, num):
    # tcar为需卸载车辆， scar为服务车辆
    num_car = 21
    num_scar = int(num_car * 1 / 3)
    num_tcar = num_car - num_scar

    # 固定任务数量为10
    num_task = 10

    env = ENV_DDPG_LOCAL(num_car, num_tcar, num_scar, num_task)
    n_actions = 2
    n_state = 48
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    score_record = []
    score_record_step = []
    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    time_record = []
    time_record_step = []
    print('服务车辆数量： {}'.format(num_scar))

    for i in range(episode):
        score = 0
        obs = env.get_init_state()
        done = False

        while not done:
            act = MECSnet.choose_action(obs)
            new_state, reward, done = env.step(act)
            MECSnet.remember(obs, act, reward, new_state, int(done))
            MECSnet.learn()
            score += reward
            obs = new_state

        episode_record.append(i)
        cost_record.append(-score)
        time_record.append(env.t_all)
        # print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            # cost_record_step.append(np.mean(cost_record))
            time_record_step.append(np.mean(time_record))

    df = pd.DataFrame({"Episode": episode_record_step, "Time": time_record_step}).set_index('Episode')
    df.to_excel("T_alltime/episode_alltime_ddpg_local/episode_alltime_ddpg_local_{}.xlsx".format(num))

    plt.figure()
    x_data = range(len(time_record_step))
    plt.plot(x_data, time_record_step)

    plt.show()

def ddpg_nogan(episode, step, num):
    # tcar为需卸载车辆， scar为服务车辆
    num_car = 21
    num_scar = int(num_car * 1 / 3)
    num_tcar = num_car - num_scar

    # 固定任务数量为10
    num_task = 10

    env = ENV_DDPG_NOGAN(num_car, num_tcar, num_scar, num_task)
    n_actions = 2
    n_state = 41
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    score_record = []
    score_record_step = []
    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    time_record = []
    time_record_step = []
    print('服务车辆数量： {}'.format(num_scar))

    for i in range(episode):
        score = 0
        obs = env.get_init_state()
        done = False

        while not done:
            act = MECSnet.choose_action(obs)
            new_state, reward, done = env.step(act)
            MECSnet.remember(obs, act, reward, new_state, int(done))
            MECSnet.learn()
            score += reward
            obs = new_state

        episode_record.append(i)
        cost_record.append(-score)
        time_record.append(env.t_all)
        # print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            # cost_record_step.append(np.mean(cost_record))
            time_record_step.append(np.mean(time_record))

    df = pd.DataFrame({"Episode": episode_record_step, "Time": time_record_step}).set_index('Episode')
    df.to_excel("T_alltime/episode_alltime_ddpg_nogan/episode_alltime_ddpg_nogan_{}.xlsx".format(num))

    plt.figure()
    x_data = range(len(time_record_step))
    plt.plot(x_data, time_record_step)

    plt.show()

def ddpg_nocache(episode, step, num):
    # tcar为需卸载车辆， scar为服务车辆
    num_car = 21
    num_scar = int(num_car * 1 / 3)
    num_tcar = num_car - num_scar

    # 固定任务数量为10
    num_task = 10

    env = ENV_DDPG_NOCACHE(num_car, num_tcar, num_scar, num_task)
    n_actions = 2
    n_state = 45
    MECSnet = Agent(alpha=0.0004, beta=0.004, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=500,
                    layer2_size=300, n_actions=n_actions)

    score_record = []
    score_record_step = []
    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    time_record = []
    time_record_step = []
    print('服务车辆数量： {}'.format(num_scar))

    for i in range(episode):
        score = 0
        obs = env.get_init_state()
        done = False

        while not done:
            act = MECSnet.choose_action(obs)
            new_state, reward, done = env.step(act)
            MECSnet.remember(obs, act, reward, new_state, int(done))
            MECSnet.learn()
            score += reward
            obs = new_state

        episode_record.append(i)
        cost_record.append(-score)
        time_record.append(env.t_all)
        # print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        count_record.append(1 - env.count_wrong / num_task)
        if i % step == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            # cost_record_step.append(np.mean(cost_record))
            time_record_step.append(np.mean(time_record))

    df = pd.DataFrame({"Episode": episode_record_step, "Time": time_record_step}).set_index('Episode')
    df.to_excel("T_alltime/episode_alltime_ddpg_nocache/episode_alltime_ddpg_nocache_{}.xlsx".format(num))

    plt.figure()
    x_data = range(len(time_record_step))
    plt.plot(x_data, time_record_step)

    plt.show()

if __name__ == '__main__':
    episode = 100
    step = 20
    num = 1
    ddpg(episode, step, num)
    ddpg_local(episode, step, num)
    ddpg_nogan(episode, step, num)
    ddpg_nocache(episode, step, num)