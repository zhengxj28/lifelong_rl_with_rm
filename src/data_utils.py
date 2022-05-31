import numpy as np


def smooth(array, weight):
    s_array = np.zeros(array.shape)
    s_array[0] = array[0]
    for i in range(1, array.shape[0]):
        s_array[i] = weight * s_array[i - 1] + (1 - weight) * array[i]
    return s_array


def reward2step(plot_reward, max_episode_length):
    # plot_reward.shape=[steps_num]
    plot_steps = max_episode_length * np.ones(plot_reward.shape)
    cum_rewards = plot_reward[max_episode_length:] - plot_reward[:-max_episode_length]
    plot_steps[max_episode_length:] = max_episode_length / cum_rewards
    plot_steps[np.isinf(plot_steps)] = max_episode_length
    return plot_steps


def data2median(data):
    # data.shape = [x_num, y_num]
    plot_25 = np.percentile(data, q=25, axis=0)
    plot_50 = np.median(data, axis=0)
    plot_75 = np.percentile(data, q=75, axis=0)
    return plot_25, plot_50, plot_75

def data2ave(data):
    # data.shape = [x_num, y_num]
    plot_average = np.average(data, axis=0)
    plot_std = np.std(data,axis=0)

