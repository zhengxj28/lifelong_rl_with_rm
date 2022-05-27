import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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


def plot_eval_transfer(title, data_name, alg_list, plot_task_index, to_steps=True, save_fig=True,
                       use_normalize=True, max_episode_length=200, directory="data2/"):
    weight = 0.999
    plot_list = []
    legend_list = ["QRM", "QRM+RS", "Average Composition", "Max Composition", "Left Composition", "Right Composition"]
    for alg_name in alg_list:
        if use_normalize:
            plot_list.append(np.load(directory + data_name + alg_name + "norm.npy"))
        else:
            plot_list.append(np.load(directory + data_name + alg_name + ".npy"))
    # plot_list[i].shape=[len(tasks),repeated_test_times,steps_num]
    color_list = ["blue", "purple", "red", "yellow", "green", "hotpink"]
    # plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plt.title(title)
    if to_steps:
        plt.ylabel("Steps to Complete Task")
    else:
        plt.ylabel("Cumulative Rewards")
    plt.xlabel("Training Steps")
    for algorithm_i in range(len(plot_list)):
        plot_all_tasks = plot_list[algorithm_i][plot_task_index, :, :]
        plot_tasks_num, repeated_test_times, steps_num = plot_all_tasks.shape
        if to_steps:
            plt.axis([0, steps_num, 0, max_episode_length + 10])
            for task_i in range(plot_tasks_num):
                for n in range(repeated_test_times):
                    plot_all_tasks[task_i, n, :] = reward2step(plot_all_tasks[task_i, n, :], max_episode_length)
        plot_i = np.average(plot_all_tasks, axis=0)  # plot_i.shape=[repeated_test_times,steps_num]
        # plot_std = np.std(plot_i, axis=0)
        plot_ave = np.average(plot_i, axis=0)
        plot_up = np.max(plot_i, axis=0)
        plot_down = np.min(plot_i, axis=0)
        if to_steps:
            plot_ave = smooth(plot_ave, weight)
            plot_up = smooth(plot_up, weight)
            plot_down = smooth(plot_down, weight)
        linewidth = 2
        x = np.linspace(0, steps_num - 1, steps_num)
        ax.plot(plot_ave, color=color_list[algorithm_i], linewidth=linewidth, label=legend_list[algorithm_i])
        ax.fill_between(x, plot_down, plot_up, color=color_list[algorithm_i], alpha=0.1)
        # ax.legend(bbox_to_anchor=(0.5, 1.06), loc="lower center", ncol=len(plot_list))

    axins = inset_axes(ax, width="30%", height="60%", loc='lower left',
                       bbox_to_anchor=(0.65, 0.35, 1, 1),
                       bbox_transform=ax.transAxes)

    for algorithm_i in range(len(plot_list)):
        plot_all_tasks = plot_list[algorithm_i][plot_task_index, :, :]
        plot_tasks_num, repeated_test_times, steps_num = plot_all_tasks.shape
        if to_steps:
            plt.axis([0, steps_num, 0, max_episode_length + 10])
            for task_i in range(plot_tasks_num):
                for n in range(repeated_test_times):
                    plot_all_tasks[task_i, n, :] = reward2step(plot_all_tasks[task_i, n, :], max_episode_length)
        plot_i = np.average(plot_all_tasks, axis=0)  # plot_i.shape=[repeated_test_times,steps_num]
        # plot_std = np.std(plot_i, axis=0)
        plot_ave = np.average(plot_i, axis=0)
        plot_up = np.max(plot_i, axis=0)
        plot_down = np.min(plot_i, axis=0)
        if to_steps:
            plot_ave = smooth(plot_ave, weight)
            plot_up = smooth(plot_up, weight)
            plot_down = smooth(plot_down, weight)
        linewidth = 2
        x = np.linspace(0, steps_num - 1, steps_num)

        axins.set_xlim(0.1 * steps_num, 0.2 * steps_num)
        axins.set_ylim(60, 100)

        axins.plot(plot_ave, color=color_list[algorithm_i], linewidth=linewidth)
        axins.fill_between(x, plot_down, plot_up, color=color_list[algorithm_i], alpha=0.1)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k', lw=1)

    if save_fig: plt.savefig(
        r"E:/Machine Learning/RL_implementation/Reward_Machines/RMTL_6/Figure/" + data_name + '.png')
    plt.show()


def plot_op_laws(title, data_name, data_index,
                 to_steps=True, save_fig=True, use_normalize=True, directory="data2/", max_episode_length=200):
    legend_list = ["QRM", "QRM+RS", "Best Representation", "Other Representation(s)"]
    plot_qrm = np.load(directory + data_name + "_" + str(data_index[0]) + "QRMnorm.npy")
    plot_qrmrs = np.load(directory + data_name + "_" + str(data_index[0]) + "QRMrsnorm.npy")
    plot_dict = {}
    for data_i in data_index:
        plot_dict[data_i] = np.load(directory + data_name + "_" + str(data_i) + "TQRMnorm.npy")
    # plot_dict[i].shape=[len(tasks),repeated_test_times,steps_num]
    color_list = ["blue", "purple", "red", "green", "green", "hotpink", "chocolate"]

    plt.figure(figsize=(8, 4))
    plt.clf()
    plt.title(title)
    plt.ylabel("Steps to Complete Task") if to_steps else plt.ylabel("Cumulative Rewards")
    plt.xlabel("Training Steps")
    for curve_i in range(2 + len(plot_dict)):
        if curve_i == 0:
            plot_i = plot_qrm[1]  # 1 is the target task
        elif curve_i == 1:
            plot_i = plot_qrmrs[1]
        else:
            plot_i = plot_dict[data_index[curve_i - 2]][1]
        repeated_test_times, steps_num = plot_i.shape
        if to_steps:
            plt.axis([0, steps_num, 0, max_episode_length + 10])
            for n in range(repeated_test_times):
                plot_i[n] = reward2step(plot_i[n], max_episode_length)
            plot_ave = np.average(plot_i, axis=0)
            plot_up = np.max(plot_i, axis=0)
            plot_down = np.min(plot_i, axis=0)
            if to_steps:
                weight = 0.9999
                plot_ave = smooth(plot_ave, weight)
                plot_up = smooth(plot_up, weight)
                plot_down = smooth(plot_down, weight)
            x = np.linspace(0, steps_num - 1, steps_num)
            plt.fill_between(x, plot_down, plot_up, color=color_list[curve_i], alpha=0.1)
            linewidth = 2
            plt.plot(plot_ave, color=color_list[curve_i], linewidth=linewidth, label=legend_list[curve_i])
            plt.legend(bbox_to_anchor=(0.5, 1.06), loc="lower center", ncol=2 + len(plot_dict))
    if save_fig: plt.savefig(
        r"/Figure/" + data_name + '.png')
    plt.show()


def plot_lifelong(title, data_name, alg_list, plot_task_index,
                  to_steps=True, save_fig=True, use_normalize=False, directory="data2/", max_episode_length=200):
    plot_list = []
    # legend_list=alg_list
    legend_list = ["QRM", "QRM+RS", "LSRM", "LSRM+RS"]
    for alg_name in alg_list:
        # if use_normalize and alg_name not in ["QRM","QRMrs"]:
        #     plot_list.append(np.load(directory + data_name + alg_name + "norm.npy"))
        if use_normalize:
            plot_list.append(np.load(directory + data_name + alg_name + "norm.npy"))
        else:
            plot_list.append(np.load(directory + data_name + alg_name + ".npy"))
    # plot_list[i].shape=[len(tasks),repeated_test_times,steps_num]
    subplot_row, subplot_col, subplot_index = 1, len(plot_task_index), 0
    tasks_num, total_test_times, steps_num = plot_list[0].shape
    color_list = ["blue", "purple", "red", "green", "green", "hotpink", "chocolate"]

    plt.figure(figsize=(6 * subplot_col, 4))
    plt.clf()
    for task_i in range(tasks_num):
        if task_i not in plot_task_index: continue
        subplot_index += 1
        plt.subplot(subplot_row, subplot_col, subplot_index)
        plt.title("Phase " + str(task_i + 1))
        if subplot_index == 1:
            plt.ylabel("Steps to Complete Task") if to_steps else plt.ylabel("Cumulative Rewards")
        if subplot_index == subplot_col // 2 + 1: plt.xlabel("Training Steps")
        for algorithm_i in range(len(plot_list)):
            plot_i = plot_list[algorithm_i][task_i, :, :]
            repeated_test_times, steps_num = plot_i.shape
            if to_steps:
                plt.axis([0, steps_num, 0, max_episode_length + 10])
                for n in range(repeated_test_times):
                    plot_i[n] = reward2step(plot_i[n], max_episode_length)
            plot_ave = np.average(plot_i, axis=0)
            plot_up = np.max(plot_i, axis=0)
            plot_down = np.min(plot_i, axis=0)
            if to_steps:
                weight = 0.999
                plot_ave = smooth(plot_ave, weight)
                plot_up = smooth(plot_up, weight)
                plot_down = smooth(plot_down, weight)
            x = np.linspace(0, steps_num - 1, steps_num)
            plt.fill_between(x, plot_down, plot_up, color=color_list[algorithm_i], alpha=0.1)
            linewidth = 2
            if subplot_index == subplot_col // 2 + 1:
                plt.plot(plot_ave,
                         color=color_list[algorithm_i], linewidth=linewidth, label=legend_list[algorithm_i])
            else:
                plt.plot(plot_ave, color=color_list[algorithm_i], linewidth=linewidth)
        if subplot_index == subplot_col // 2 + 1:
            plt.legend(bbox_to_anchor=(0.5, 1.06), loc="lower center", ncol=len(plot_list))
    if save_fig: plt.savefig(
        r"Figure/" + data_name + '.png')
    plt.show()
