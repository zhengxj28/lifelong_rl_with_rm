import numpy as np


def average_transfer(Q1, Q2):
    return (Q1 + Q2) / 2


def max_transfer(Q1, Q2):
    state_num, action_num = Q1.shape
    Q = np.zeros([state_num, action_num])
    Q1_max = np.max(Q1, axis=1)
    Q2_max = np.max(Q2, axis=1)
    for s in range(state_num):
        if Q1_max[s] > Q2_max[s]:
            Q[s, :] = Q1[s, :]
        else:
            Q[s, :] = Q2[s, :]
    return Q


def left_transfer(Q1, Q2):
    return Q1.copy()


def right_transfer(Q1, Q2):
    return Q2.copy()


######## boolean task algebra methods ##############
def boolean_and(Q1, Q2):
    # note that Q1.shape=Q2.shape=[goal_num, state_num, action_num]
    return np.minimum(Q1, Q2)


def boolean_or(Q1, Q2):
    return np.maximum(Q1, Q2)
