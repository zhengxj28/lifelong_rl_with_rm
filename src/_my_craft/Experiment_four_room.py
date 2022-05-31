'''
Examine the effect of LLRM model in MineCraft
'''

import os, time, sys, random

sys.path.append("..")
sys.path.append("../..")
from src._my_lifelong import *
from src._my_plot_assistant import *
from src.params import Parameters

if __name__ == '__main__':
    from worlds.craft_world import *
    import matplotlib.pyplot as plt

    is_plot = 1
    if is_plot:
        plot_lifelong(title="Four Room",
                      directory="four_room",
                      alg_list=["QRM", "QRMrs", "TQRMbest", "TQRMworst", "boolean"],
                      legend_list=["QRM", "QRM+RS", "LSRM-best", "LSRM-worst", "Boolean"],
                      steps_num=50000,
                      smooth_fac=0.9,
                      plot_task_index=[0,1,2],
                      to_steps=True,
                      save_fig=True,
                      use_normalize=True,
                      max_episode_length=500)
    else:
        map_i = 0
        param = CraftWorldParams(
            file_map="maps/four_room.map",
            use_tabular_representation=True,
            consider_night=False,
            movement_noise=0)
        craft_env = CraftWorld(param)
        a = ('eventually', 'a')
        b = ('eventually', 'b')
        c = ('eventually', 'c')
        d = ('eventually', 'd')
        # e = ('eventually', 'e')
        # f = ('eventually', 'f')
        tasks = [[a, b, ('and', c, d)],
                 [('and', a, b), ('or', c, d)],
                 [('then', ('and', a, b), ('and', c, d))],
                 ]
        repeated_test_times = 20
        propositions = ['a', 'b', 'c', 'd', ]
        label_set = ['a', 'b', 'c', 'd', '']

        params = Parameters(steps_num=400000,
                            gamma=0.9,
                            alpha=1.0,
                            epsilon=0.1,
                            max_episode_length=500,
                            use_normalize=True)

        # for algorithm in ["QRM", "QRMrs"]:
        # for algorithm in [ "TQRMworst", "TQRMbest"]:
        for algorithm in ["boolean"]:
            print(algorithm)
            run_lifelong(tasks,
                         repeated_test_times=repeated_test_times,
                         env=craft_env,
                         propositions=propositions,
                         label_set=label_set,
                         params=params,
                         algorithm=algorithm,
                         save_data=True,
                         directory="four_room")
