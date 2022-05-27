'''
Examine the effect of LLRM model in MineCraft
'''
import numpy as np

from src._my_lifelong import *
from src._my_plot_assistant import *


if __name__ == '__main__':
    from worlds.craft_world import *
    import matplotlib.pyplot as plt

    is_plot = True
    if is_plot:
        plot_op_laws(title="Different Representations of \"Making Bed\"",
                data_name="E4",
                data_index=[1,2],
                to_steps=True,
                save_fig=False,
                use_normalize=True)
    else:
        map_i = 0
        param = CraftWorldParams(
            file_map="maps/map_" + str(map_i) + ".map",
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
        for experiment_i in [1,2,3]:
            if experiment_i==1:
                source_tasks=[('then',a,b),c,d]
                target_tasks=[('then', ('and', ('then', a, b), d), c)]
            if experiment_i==2:
                source_tasks = [('then', a, b), c, d]
                target_tasks = [('and',('then',('then',a,b),c),('then',d,c))]
            if experiment_i==3:
                source_tasks = [('then', a, b), c, d]
                target_tasks = [('and',('then',a,('then',b,c)),('then',d,c))]

            tasks = [source_tasks,
                     target_tasks,
                     ]
            steps_num = 400000
            repeated_test_times = 20
            propositions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
            label_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', '']
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
                             env=craft_env,
                             propositions=propositions,
                             label_set=label_set,
                             gamma=gamma,
                             alpha=alpha,
                             epsilon=epsilon,
                             max_episode_length=max_episode_length,
                             algorithm=algorithm,
                             save_data=save_data,
                             data_name=data_name)
