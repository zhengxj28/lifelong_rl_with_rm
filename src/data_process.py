import numpy as np
import os

def reward2step(plot_reward, max_episode_length):
    # plot_reward.shape=[steps_num]
    plot_steps = max_episode_length * np.ones(plot_reward.shape)
    cum_rewards = plot_reward[max_episode_length:] - plot_reward[:-max_episode_length]
    plot_steps[max_episode_length:] = max_episode_length / cum_rewards
    plot_steps[np.isinf(plot_steps)] = max_episode_length
    return plot_steps

# def reward2step(plot_reward, max_episode_length):
#     # plot_reward.shape=[tasks_num, trails_num, steps_num]
#     tasks_num, trails_num, steps_num = plot_reward.shape
#     plot_steps = max_episode_length * np.ones(plot_reward.shape)
#
#     for task_id in range(tasks_num):
#         for trail_id in range(trails_num):
#             cum_rewards = plot_reward[task_id, trail_id, max_episode_length:] - plot_reward[task_id, trail_id, :-max_episode_length]
#             plot_steps[task_id, trail_id, max_episode_length:] = max_episode_length / cum_rewards
#             one_dim_steps = plot_steps[task_id,trail_id,:]
#             one_dim_steps[np.isinf(one_dim_steps)] = max_episode_length
#             plot_steps[task_id,trail_id, :] = one_dim_steps
#     return plot_steps


if __name__ == '__main__':
    projectDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    data_path = os.path.join(projectDir, 'lifelong_rl_with_rm_data', "minecraft", "reward(2021)")
    zip_data_path = os.path.join(projectDir, 'lifelong_rl_with_rm_data', "minecraft", "steps(2021)")

    for algorithm in ["QRM", "QRMrs", "TQRMbest", "TQRMworst"]:
        data_reward = np.load(os.path.join(data_path, algorithm+"norm.npy"))
        tasks_num, trails_num, steps_num = data_reward.shape
        data_step = np.zeros(data_reward.shape)
        for task_id in range(tasks_num):
            for trail_id in range(trails_num):
                data_step[task_id,trail_id] = reward2step(data_reward[task_id,trail_id], 500)
        # zip_data = data_step[:,:,::100]
        zip_data = data_step
        np.save(os.path.join(zip_data_path, algorithm+"norm.npy"), zip_data)
