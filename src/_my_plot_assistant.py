import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

from src.data_utils import *


def plot_eval_transfer(title, data_name, alg_list, plot_task_index, to_steps=True, save_fig=True,
                       use_normalize=True, max_episode_length=200, directory="data/"):
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
        r"E:/Machine Learning/RL_implementation/Reward_Machines/RMTL_6/Figure/" + data_name + '.pdf')
    plt.show()


def plot_op_laws(title, data_name, data_index,
                 to_steps=True, save_fig=True, use_normalize=True, directory="data/", max_episode_length=200):
    legend_list = ["QRM", "QRM+RS", "Best Representation", "Other Representation(s)"]
    projectDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
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
    if save_fig: plt.savefig(os.path.join(projectDir, "Figure", directory + '.pdf'))
    plt.show()


def plot_lifelong(title, directory, alg_list, legend_list, steps_num, smooth_fac, plot_task_index,
                  to_steps=True, save_fig=True, use_normalize=False, max_episode_length=200):
    from src._my_lifelong import test_steps
    plot_list = []
    projectDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    if to_steps:
        data_path = os.path.join(projectDir, 'lifelong_rl_with_rm_data', directory, "steps")
    else:
        data_path = os.path.join(projectDir, 'lifelong_rl_with_rm_data', directory, "reward")
    for alg_name in alg_list:
        # if use_normalize and alg_name not in ["QRM","QRMrs"]:
        #     plot_list.append(np.load(directory + data_name + alg_name + "norm.npy"))
        if use_normalize:
            plot_list.append(np.load(os.path.join(data_path, alg_name + "norm.npy")))
        else:
            plot_list.append(np.load(os.path.join(data_path, alg_name + ".npy")))
    # plot_list[i].shape=[len(tasks),repeated_test_times,x_num]
    subplot_row, subplot_col, subplot_index = 1, len(plot_task_index), 0
    tasks_num, total_test_times, x_num = plot_list[0].shape
    color_list = ["blue", "purple", "yellow", "red", "green", "hotpink", "chocolate"]

    fig = plt.figure(figsize=(6 * subplot_col, 4))
    plt.clf()
    for task_i in range(tasks_num):
        if task_i not in plot_task_index: continue
        subplot_index += 1
        ax = plt.subplot(subplot_row, subplot_col, subplot_index)
        plt.title("Phase " + str(task_i + 1))
        if subplot_index == 1:
            plt.ylabel("Steps to Complete Task") if to_steps else plt.ylabel("Cumulative Rewards")
        if subplot_index == subplot_col // 2 + 1: plt.xlabel("Training Steps")
        for algorithm_i in range(len(plot_list)):
            plot_i = plot_list[algorithm_i][task_i, :, :int(steps_num/test_steps)]
            repeated_test_times, x_num = plot_i.shape
            plot_down, plot_ave, plot_up = data2median(plot_i)
            if to_steps:
                weight = smooth_fac
                plot_ave = smooth(plot_ave, weight)
                plot_up = smooth(plot_up, weight)
                plot_down = smooth(plot_down, weight)
            x = test_steps * np.linspace(0, x_num - 1, x_num)
            plt.fill_between(x, plot_down, plot_up, color=color_list[algorithm_i], alpha=0.1)
            linewidth = 4 if alg_list[algorithm_i]=="TQRMbest" else 2
            if subplot_index == subplot_col // 2 + 1:
                plt.plot(x, plot_ave,
                         color=color_list[algorithm_i], linewidth=linewidth, label=legend_list[algorithm_i])
            else:
                plt.plot(x, plot_ave, color=color_list[algorithm_i], linewidth=linewidth)

        if steps_num > 100000:
            ticks = np.append(x[::500], np.max(x)+test_steps) #[0,100000,200000,300000,400000]
        else:
            ticks = np.append(x[::100], np.max(x)+test_steps)  #[0, 10000, 20000, 30000]
        labels = [str(int(tick / 1000)) + 'k' for tick in ticks]
        plt.setp(ax, xticks=ticks, xticklabels=labels)

        if subplot_index == subplot_col // 2 + 1:
            plt.legend(bbox_to_anchor=(0.5, 1.05), loc="lower center", ncol=len(plot_list))
    plt.subplots_adjust(left=0.05,
                        bottom=0.12,
                        right=0.98,
                        top=0.88,
                        wspace=0.09,
                        hspace=0.2)
    # fig.tight_layout()
    if save_fig: plt.savefig(os.path.join(projectDir, "Figure", directory + '.pdf'))
    plt.show()


def plot_lifelong2(title, directory, alg_list, legend_list, steps_num, smooth_fac, plot_task_index,
                  to_steps=True, save_fig=True, use_normalize=False, max_episode_length=200):
    from src._my_lifelong import test_steps
    plot_list = []
    projectDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    if to_steps:
        data_path = os.path.join(projectDir, 'lifelong_rl_with_rm_data', directory, "steps(2021)")
    else:
        data_path = os.path.join(projectDir, 'lifelong_rl_with_rm_data', directory, "reward")
    for alg_name in alg_list:
        # if use_normalize and alg_name not in ["QRM","QRMrs"]:
        #     plot_list.append(np.load(directory + data_name + alg_name + "norm.npy"))
        if use_normalize:
            plot_list.append(np.load(os.path.join(data_path, alg_name + "norm.npy")))
        else:
            plot_list.append(np.load(os.path.join(data_path, alg_name + ".npy")))
    # plot_list[i].shape=[len(tasks),repeated_test_times,x_num]
    subplot_row, subplot_col, subplot_index = 1, len(plot_task_index), 0
    tasks_num, total_test_times, x_num = plot_list[0].shape
    color_list = ["blue", "purple", "red", "green", "hotpink", "chocolate"]

    fig = plt.figure(figsize=(6 * subplot_col, 4))
    plt.clf()
    for task_i in range(tasks_num):
        if task_i not in plot_task_index: continue
        subplot_index += 1
        ax = plt.subplot(subplot_row, subplot_col, subplot_index)
        plt.title("Phase " + str(task_i + 1))
        if subplot_index == 1:
            plt.ylabel("Steps to Complete Task") if to_steps else plt.ylabel("Cumulative Rewards")
        if subplot_index == subplot_col // 2 + 1: plt.xlabel("Training Steps")
        for algorithm_i in range(len(plot_list)):
            plot_i = plot_list[algorithm_i][task_i, :, :int(steps_num/test_steps)]
            repeated_test_times, x_num = plot_i.shape
            plot_down, plot_ave, plot_up = data2median(plot_i)
            if to_steps:
                weight = smooth_fac
                plot_ave = smooth(plot_ave, weight)
                plot_up = smooth(plot_up, weight)
                plot_down = smooth(plot_down, weight)
            x = test_steps * np.linspace(0, x_num - 1, x_num)
            plt.fill_between(x, plot_down, plot_up, color=color_list[algorithm_i], alpha=0.1)
            linewidth = 2 if alg_list[algorithm_i]=="TQRMbest" else 1
            if subplot_index == subplot_col // 2 + 1:
                plt.plot(x, plot_ave,
                         color=color_list[algorithm_i], linewidth=linewidth, label=legend_list[algorithm_i])
            else:
                plt.plot(x, plot_ave, color=color_list[algorithm_i], linewidth=linewidth)

        if steps_num > 100000:
            ticks = np.append(x[::500], np.max(x)+test_steps) #[0,100000,200000,300000,400000]
        else:
            ticks = np.append(x[::100], np.max(x)+test_steps)  #[0, 10000, 20000, 30000]
        labels = [str(int(tick / 1000)) + 'k' for tick in ticks]
        plt.setp(ax, xticks=ticks, xticklabels=labels)

        if subplot_index == subplot_col // 2 + 1:
            plt.legend(bbox_to_anchor=(0.5, 1.05), loc="lower center", ncol=len(plot_list))
    plt.subplots_adjust(left=0.05,
                        bottom=0.12,
                        right=0.98,
                        top=0.88,
                        wspace=0.09,
                        hspace=0.2)
    # fig.tight_layout()
    if save_fig: plt.savefig(os.path.join(projectDir, "Figure", directory + '.pdf'))
    plt.show()

def plot_alpha(directory, alpha_list, legend_list, steps_num, smooth_fac, plot_task_index,
                  to_steps=True, save_fig=True, use_normalize=False):
    from src._my_lifelong import test_steps
    plot_list = []
    projectDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

    for i in range(len(alpha_list)):
        data_path = os.path.join(projectDir, 'lifelong_rl_with_rm_data')
        if use_normalize:
            plot_list.append(np.load(os.path.join(data_path, directory+"_"+str(alpha_list[i]),
                                                  "steps", "QRMnorm.npy")))
        else:
            plot_list.append(np.load(os.path.join(data_path, directory +"_"+ str(alpha_list[i]),
                                                  "steps", "QRM.npy")))
    # plot_list[i].shape=[len(tasks),repeated_test_times,x_num]
    subplot_row, subplot_col, subplot_index = 1, len(plot_task_index), 0
    tasks_num, total_test_times, x_num = plot_list[0].shape
    color_list = ["blue", "purple", "yellow", "red", "green", "hotpink", "chocolate"]

    fig = plt.figure(figsize=(6 * subplot_col, 4))
    plt.clf()
    for phase_i in range(tasks_num):
        if phase_i not in plot_task_index: continue
        subplot_index += 1
        ax = plt.subplot(subplot_row, subplot_col, subplot_index)
        # plt.title("Phase " + str(task_i + 1))
        plt.ylabel("Steps to Complete Task")
        plt.xlabel("Training Steps")
        for alpha_i in range(len(plot_list)):
            plot_i = plot_list[alpha_i][phase_i, :, :int(steps_num/test_steps)]
            repeated_test_times, x_num = plot_i.shape
            plot_down, plot_ave, plot_up = data2median(plot_i)
            if to_steps:
                weight = smooth_fac
                plot_ave = smooth(plot_ave, weight)
                plot_up = smooth(plot_up, weight)
                plot_down = smooth(plot_down, weight)
            x = test_steps * np.linspace(0, x_num - 1, x_num)
            plt.fill_between(x, plot_down, plot_up, color=color_list[alpha_i], alpha=0.1)
            linewidth = 2
            if subplot_index == subplot_col // 2 + 1:
                plt.plot(x, plot_ave,
                         color=color_list[alpha_i], linewidth=linewidth, label=legend_list[alpha_i])
            else:
                plt.plot(x, plot_ave, color=color_list[alpha_i], linewidth=linewidth)

        if steps_num > 100000:
            ticks = np.append(x[::500], np.max(x)+test_steps) #[0,100000,200000,300000,400000]
        else:
            ticks = np.append(x[::100], np.max(x)+test_steps)  #[0, 10000, 20000, 30000]
        labels = [str(int(tick / 1000)) + 'k' for tick in ticks]
        plt.setp(ax, xticks=ticks, xticklabels=labels)

        if subplot_index == subplot_col // 2 + 1:
            plt.legend(bbox_to_anchor=(0.5, 1.05), loc="lower center", ncol=len(plot_list))
    plt.subplots_adjust(left=0.13,
                        bottom=0.12,
                        right=0.98,
                        top=0.88,
                        wspace=0.09,
                        hspace=0.2)
    # fig.tight_layout()
    if save_fig: plt.savefig(os.path.join(projectDir, "Figure", directory + 'alpha.pdf'))
    plt.show()