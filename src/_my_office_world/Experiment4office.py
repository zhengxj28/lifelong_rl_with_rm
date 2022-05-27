'''
Examine the effect of LLRM model in MineCraft
'''
import numpy as np

from src._my_lifelong import *
from src._my_plot_assistant import *


if __name__ == '__main__':
    from worlds.office_world import *
    import matplotlib.pyplot as plt

    is_plot = True
    if is_plot:
        plot_op_laws(title="Different Representations of \"Delivering Coffee and Mail to Office\"",
                data_name="E4",
                data_index=[1,2],
                to_steps=True,
                save_fig=False,
                use_normalize=True)
    else:
        map_i = 0
        param = OfficeWorldParams()
        office_env = OfficeWorld(param)
        A = ('until', ('not', 'n'), 'a')
        B = ('until', ('not', 'n'), 'b')
        C = ('until', ('not', 'n'), 'c')
        D = ('until', ('not', 'n'), 'd')
        c = ('until', ('not', 'n'), 'f')
        m = ('until', ('not', 'n'), 'e')
        o = ('until', ('not', 'n'), 'g')
        for experiment_i in [1,2]:
            if experiment_i==1:
                source_tasks=[('then',c,o),('then',m,o)]
                target_tasks=[('and',('then',c,o),('then',m,o))]
            if experiment_i==2:
                source_tasks = [('then',c,o),('then',m,o)]
                target_tasks = [('then',('and',c,m),o)]
            tasks = [source_tasks,
                     target_tasks,
                     ]
            steps_num = 30000
            repeated_test_times = 20
            propositions = ['a', 'b', 'c', 'd', 'f', 'e', 'g', 'n']
            label_set = ['a', 'b', 'c', 'd', 'f', 'e', 'g', 'n', '']
            gamma = 0.9
            alpha = 1.0
            epsilon = 0.1
            max_episode_length = 200
            save_data = True
            data_name = "E4_" + str(experiment_i)
            for algorithm in ["QRM", "QRMrs", "TQRM"]:
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
                             data_name=data_name)
