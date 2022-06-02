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
    from worlds.office_world import *
    import matplotlib.pyplot as plt

    is_plot = 1
    if is_plot:
        plot_lifelong(title="Experiment3",
                      directory="office",
                      alg_list=["QRM", "QRMrs", "equiv","TQRMbest", "TQRMworst", "boolean"],
                      legend_list=["QRM", "QRM+RS","EQUIV", "LSRM-best", "LSRM-worst", "LOGICAL"],
                      steps_num=30000,
                      smooth_fac=0.9,
                      plot_task_index=[i for i in range(3)],
                      to_steps=True,
                      save_fig=True,
                      use_normalize=True)
    else:
        param = OfficeWorldParams()
        office_env = OfficeWorld(param)
        A = ('until', ('not', 'n'), 'a')
        B = ('until', ('not', 'n'), 'b')
        C = ('until', ('not', 'n'), 'c')
        D = ('until', ('not', 'n'), 'd')
        c = ('until', ('not', 'n'), 'f')
        m = ('until', ('not', 'n'), 'e')
        o = ('until', ('not', 'n'), 'g')
        # tasks = [[('then', c, A), ('then', m, B)],
        #          [('then', c, o), ('then', m, o)],
        #          [('and', ('then', c, o), ('then', m, o)), ('then', B, A)],
        #          ]
        tasks = [[('then', c, A), ('then', m, B)],
                 [('then', c, o), ('then', m, o)],
                 [('then', ('and', c, m), o), ('then', B, A)],
                 ]
        repeated_test_times = 20
        propositions = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        label_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', '']
        params = Parameters(steps_num=30000,
                            gamma=0.9,
                            alpha=1.0,
                            epsilon=0.1,
                            max_episode_length=200,
                            use_normalize=True)
        save_data = True
        # for algorithm in ["QRM", "QRMrs"]:
        # for algorithm in ["TQRMworst", "TQRMbest"]:
        # for algorithm in ["boolean"]:
        for algorithm in ["TQRMworst2", "equiv"]:
            print(algorithm)
            run_lifelong(tasks,
                         repeated_test_times=repeated_test_times,
                         env=office_env,
                         propositions=propositions,
                         label_set=label_set,
                         params=params,
                         algorithm=algorithm,
                         save_data=save_data,
                         directory="office")
