'''
Examine the effect of LLRM model in MineCraft
'''

from src._my_lifelong import *
from src._my_plot_assistant import *

if __name__ == '__main__':
    from worlds.craft_world import *
    import matplotlib.pyplot as plt

    data_name = "four_room"

    is_plot = False
    if is_plot:
        plot_lifelong(title="Experiment5",
                      data_name=data_name,
                      alg_list=["QRM", "QRMrs", "TQRM", "TQRMrs"],
                      plot_task_index=[0, 2, 4],
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
        tasks = [[a,b,('and',c,d)],
                 [('and',a,b), ('or', c,d)],
                 [('then',('and',a,b),('and',c,d))],
                 ]
        steps_num = 400000
        repeated_test_times = 20
        propositions = ['a', 'b', 'c', 'd', ]
        label_set = ['a', 'b', 'c', 'd', '']
        gamma = 0.9
        alpha = 1.0
        epsilon = 0.1
        max_episode_length = 500
        save_data = True
        # for algorithm in ["TQRM","TQRMrs"]:
        # for algorithm in ["advisor"]:
        for algorithm in ["QRM","QRMrs","TQRM"]:
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
