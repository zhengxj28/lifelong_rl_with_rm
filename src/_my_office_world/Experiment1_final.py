'''
Examine the effect of non-equivalent transfer methods in OfficeWorld.
'''
import numpy as np

from src._my_lifelong import *
from src._my_plot_assistant import *

if __name__ == '__main__':
    from worlds.office_world import *

    is_plot = True
    if is_plot:
        for experiment_i in [1]:
            plot_eval_transfer(title="Target Tasks Composed by \"and\"",
                          data_name="E1_final_" + str(experiment_i),
                          alg_list=["QRM", "QRMrs", "TQRMaverage", "TQRMmax", "TQRMleft",
                                    "TQRMright"],
                          plot_task_index=[1,2],
                          to_steps=True,
                          save_fig=True,
                          use_normalize=True,
                          directory="data2/")
    else:
        # for ij in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]:
        # for ij in [(4,1),(4,2),(4,3)]:
        for experiment_i in [1]:
            param = OfficeWorldParams()
            office_env = OfficeWorld(param)
            A = ('until', ('not', 'n'), 'a')
            B = ('until', ('not', 'n'), 'b')
            C = ('until', ('not', 'n'), 'c')
            D = ('until', ('not', 'n'), 'd')
            c = ('until', ('not', 'n'), 'f')
            m = ('until', ('not', 'n'), 'e')
            o = ('until', ('not', 'n'), 'g')
            t1 = ('then', c, o)
            t2 = ('then', m, o)
            t3 = ('then', B, o)
            t4 = ('then', B, C)
            t5 = ('then', o, C)
            t6 = ('then', m, D)
            t7 = ('then', c, o)
            t8 = ('then', B, C)
            t9 = ('then', m, A)
            if experiment_i == 1:
                tasks = [[t1, t2, t3, t4, t5, t6],
                         [('and', t4, t6)],[('and',t6,t4)],[('and',t1,t3)],[('and',t3,t1)]]
            if experiment_i == 2:
                tasks = [[t4, t5, t6,t7,t8,t9],
                         [('or', t4, t5)],[('or',t5,t4)],[('or',t8,t9)],[('or',t9,t8)]]
            if experiment_i == 3:
                tasks = [[t1, t2, t3, t4, t5, t6],
                         [('then', t4, t5)], [('then', t4, t6)], [('then', t5, t6)]]
            ################# lifelong learning ###################
            steps_num = 30000
            repeated_test_times = 20
            propositions = ['a', 'b', 'c', 'd', 'f', 'e', 'g', 'n']
            label_set = ['a', 'b', 'c', 'd', 'f', 'e', 'g', 'n', '']
            gamma = 0.9
            alpha = 1.0
            epsilon = 0.1
            max_episode_length = 200
            save_data = True
            data_name = "E1_final_" + str(experiment_i)
            for algorithm in ["QRM", "QRMrs", "equiv", "TQRMaverage", "TQRMmax", "TQRMleft", "TQRMright"]:
                # for algorithm in ["QRMrs",]:
                run_lifelong(tasks,
                             steps_num=steps_num,
                             repeated_test_times=repeated_test_times,
                             env=office_env,
                             propositions=propositions,
                             label_set=label_set,
                             gamma=gamma,
                             alpha=alpha,
                             epsilon=epsilon,
                             max_episode_length=max_episode_length,
                             algorithm=algorithm,
                             save_data=save_data,
                             data_name=data_name,
                             directory="data3/")
