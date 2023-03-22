'''
Examine the effect of LLRM model in MineCraft
'''

import sys

sys.path.append("..")
sys.path.append("../..")
from src._my_lifelong import *
from src._my_plot_assistant import *
from src.params import Parameters

if __name__ == '__main__':
    from worlds.craft_world import *
    import matplotlib.pyplot as plt

    data_name = "E3"

    is_plot = 1
    if is_plot:
        plot_alpha(directory="minecraft",
                   alpha_list=[0.1,0.5,1.0],
                   legend_list=["alpha=0.1","alpha=0.5","alpha=1.0"],
                   steps_num=400000,
                   smooth_fac=0.99,
                   plot_task_index=[0,],
                   to_steps=True,
                   save_fig=True,
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
        tasks = [[('then', a, b), ('then', a, c)],
                 ]
        repeated_test_times = 20

        propositions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        label_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', '']

        save_data = True
        for lr in [0.1, 0.5, 1.0]:
            params = Parameters(steps_num=400000,
                                gamma=0.9,
                                alpha=lr,
                                epsilon=0.1,
                                max_episode_length=500,
                                use_normalize=True)
            run_lifelong(tasks,
                         repeated_test_times=repeated_test_times,
                         env=craft_env,
                         propositions=propositions,
                         label_set=label_set,
                         params=params,
                         algorithm="QRM",
                         save_data=save_data,
                         directory="minecraft_" + str(lr))
