'''
Examine the effect of non-equivalent transfer methods in MineCraft
'''

from src._my_lifelong import *
from src._my_plot_assistant import *

if __name__ == '__main__':
    from worlds.craft_world import *
    import matplotlib.pyplot as plt

    # experiment_i, experiment_j = 1, 1
    is_plot = True
    if is_plot:
        for experiment_i in [3]:
            plot_eval_transfer("Target Tasks Composed by \"then\"",
                               "E2_final_" + str(experiment_i),
                               alg_list=["QRM", "QRMrs", "TQRMaverage", "TQRMmax", "TQRMleft",
                                         "TQRMright"],
                               to_steps=True,
                               use_normalize=True,
                               plot_task_index=[1, 2],
                               max_episode_length=200)
    else:
        for experiment_i in [1]:
            param = CraftWorldParams(
                file_map="maps/map_0.map",
                use_tabular_representation=True,
                consider_night=False,
                movement_noise=0)
            craft_env = CraftWorld(param)
            a = ('eventually', 'a')
            b = ('eventually', 'b')
            c = ('eventually', 'c')
            d = ('eventually', 'd')
            e = ('eventually', 'e')
            f = ('eventually', 'f')
            g = ('eventually', 'g')
            h = ('eventually', 'h')
            t1 = ('then', a, b)
            t2 = ('then', a, c)
            t3 = ('then', d, e)
            t4 = ('then', d, b)
            t5 = ('then', ('and', a, f), e)
            t6 = ('then', ('and', a, f), c)
            if experiment_i == 1:
                tasks = [[t1, t2, t3, t4, t5, t6],
                         [('and', t1, t3)], [('and', t4, t5)], [('and', t4, t6)]]
            elif experiment_i == 2:
                tasks = [[t1, t2, t3, t4, t5, t6],
                         [('or', t1, t3)], [('or', t2, t3)]]
            elif experiment_i == 3:
                tasks = [[t1, t2, t3, t4, t5, t6],
                         [('then', t2, t3)], [('then', t5, t6)]]
            steps_num = 400000
            repeated_test_times = 20
            propositions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            label_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', '']
            gamma = 0.9
            alpha = 1.0
            epsilon = 0.1
            max_episode_length = 200
            save_data = True
            data_name = "E2_final_" + str(experiment_i)
            for algorithm in ["QRM", "QRMrs", "equiv", "TQRMaverage", "TQRMmax", "TQRMleft", "TQRMright"]:
                # for algorithm in ["TQRMaverage", "TQRMmax", "TQRMleft", "TQRMright"]:
                run_lifelong(tasks,
                             steps_num=steps_num,
                             repeated_test_times=repeated_test_times,
                             env=craft_env,
                             propositions=propositions,
                             label_set=label_set,
                             gamma=gamma,
                             alpha=alpha,
                             epsilon=epsilon,
                             max_episode_length=max_episode_length,
                             algorithm=algorithm,
                             save_data=save_data,
                             data_name=data_name,
                             directory="data_final")
