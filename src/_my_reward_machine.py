# Reward Machine = DFA + reward function + Q-functions (array/network structure)
import numpy as np
from src.dfa import DFA
from src._my_network import Tabular_Q, DQN


class RewardMachine():
    def __init__(self, env, dfa, transfer=None, use_rs=False, algorithm=None):
        # <U,u0,delta_u,delta_r>
        self.env = env
        self.dfa = dfa
        self.use_rs = use_rs
        self.gamma = 0.9
        self.potentials = self.value_iteration()  # dict: {u_state: value}
        self.state_num = self.env.get_features().size  # state_num of the environment (not the RM)
        self.action_num = len(self.env.get_actions())
        self.goal_num = self.env.goal_num
        self.goal2id = self.env.goal2id
        self.label_num = len(self.dfa.label_set)
        if algorithm is not None:
            self.algorithm = algorithm
        else:
            self.algorithm = None
        self.u_num = self.dfa.non_terminal_state_num
        self.Q = self.build_Q(transfer)  # Q.shape=(u_num, self.env.state_num, self.env.action_num)
        self.max_states = 25

        self.LU_table = {}  # quick search for label->transited states
        self.LR_table = {}  # quick search for label->rewards for each states
        for l in self.dfa.label_set:
            self.LU_table[l] = np.zeros((self.max_states), dtype=int)
            self.LR_table[l] = np.zeros((self.max_states), dtype=float)

        # NOTE(about self.use_rm_matching):
        # In the Maps for the ICML paper, we included a simple graph matching approach
        # to share policies between subsets of RMs that were equivalent. We deactivated this
        # feature for the IJCAI Maps because it was too expensive and has marginal effects
        # on the performance of QRM and QRM-RS.
        self.use_rm_matching = False

    def initialize_Q(self, transfer_method=None):
        self.Q = self.build_Q(transfer_method)

    # Public methods -----------------------------------
    def build_Q(self, transfer_method=None):
        # there are many Q function for each (non terminal) state in each task
        Q = {}
        for u in self.dfa.state2ltl:
            if u not in self.dfa.terminal:
                Q[u] = self.build_one_Q()
        # if transfer_method == None:
        #     pass
        # else:
        #     transfer_function = transfer_method[0]
        #     source_Q = transfer_method[1]
        #     transfer_function(source_Q, Q)
        return Q

    def build_one_Q(self):
        # if self.use_rs: r_min,r_max=1, 1/(1-self.gamma)+2
        # else: r_min,r_max=0,1
        # r_min,r_max=0,1
        if self.env.is_discrete:
            if self.algorithm == "boolean":
                return Tabular_Q([self.goal_num, self.state_num, self.action_num])
            else:
                return Tabular_Q([self.state_num, self.action_num])
        else:
            return DQN(self.state_num, self.action_num)

    def get_initial_state(self):
        return self.dfa.ltl2state[self.dfa.formula]

    def get_next_state(self, u1, true_props):
        return self.dfa._get_next_state(u1, true_props)

        # if u1 < self.u_broken:
        #     for u2 in self.delta_u[u1]:
        #         if evaluate_dnf(self.delta_u[u1][u2], true_props):
        #             return u2
        # return self.u_broken # no transition is defined for true_props

    def get_reward(self, u1, u2, s1=0, a=0, s2=0, is_training=False):
        """
            Returns the reward associated to this transition.
            The extra reward given by RS is included only during training!
        """
        # if u2 not in self.dfa.state2ltl:
        #     return 0
        try:
            if self.dfa.state2ltl[u2] == 'True':
                r = 1
            else:
                r = 0
        except:
            r = 0
        if (self.use_rs) and (is_training) and (s1 != s2):
            rs = self.gamma * self.potentials[u2] - self.potentials[u1]
            return r + rs
        else:
            return r

    # def get_rewards_and_next_states(self, s1, a, s2, true_props, is_training):
    #     rewards = []
    #     next_states = []
    #     for u1 in self.U:
    #         u2 = self.get_next_state(u1, true_props)
    #         rewards.append(self.get_reward(u1,u2,s1,a,s2,is_training))
    #         next_states.append(u2)
    #     return rewards, next_states

    def get_state(self):  # get current state
        return self.dfa.state

    def is_terminal_state(self, u1):
        return u1 in self.dfa.terminal

    ######## reward shaping ##################
    def value_iteration(self):
        """
        Standard value iteration approach.
        We use it to compute the potentials function for the automated reward shaping
        """
        potentials = {}
        for u_state in self.dfa.state2ltl:
            potentials[u_state] = 0
        V_error = 1
        while V_error > 1e-7:
            V_error = 0
            for u1 in self.dfa.state2ltl:
                if u1 in self.dfa.terminal: continue
                q_u2 = []
                for l, u2 in self.dfa.transitions[u1].items():
                    # if u2 in self.dfa.terminal: break
                    r = self.get_reward(u1, u2, is_training=False)
                    q_u2.append(r + self.gamma * potentials[u2])
                v_new = max(q_u2)
                V_error = max([V_error, abs(v_new - potentials[u1])])
                potentials[u1] = v_new
        for u in potentials:
            potentials[u] = -potentials[u]
        # potentials[-1]=-2.0
        return potentials

    def expand(self, formula):
        new_states = self.dfa.expand_dfa(formula)
        for u in new_states:
            self.Q[u] = self.build_one_Q()
        self.potentials = self.value_iteration()  # used for reward shaping
        if not self.env.is_discrete:
            for u, lu in self.dfa.transitions.items():
                for l, u_ in lu.items():
                    self.LU_table[l][u] = u_
                    self.LR_table[l][u] = self.get_reward(u, u_)
        return new_states

    # def is_this_machine_equivalent(self, u1, rm2, u2):
    #     """
    #     return True iff
    #         this reward machine initialized at u1 is equivalent
    #         to the reward machine rm2 initialized at u2
    #     """
    #     if not self.use_rm_matching:
    #         return False
    #     return are_these_machines_equivalent(self, u1, rm2, u2)

    # def get_useful_transitions(self, u1):
    #     # This is an auxiliary method used by the HRL baseline to prune "useless" options
    #     return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]


if __name__ == '__main__':
    from worlds.office_world import *
    from worlds.water_world import *

    # param = OfficeWorldParams()
    # office_env = OfficeWorld(param)
    # A = ('until', ('not', 'n'), 'a')
    # B = ('until', ('not', 'n'), 'b')
    # C = ('until', ('not', 'n'), 'c')
    # D = ('until', ('not', 'n'), 'd')
    # c = ('until', ('not', 'n'), 'f')
    # m = ('until', ('not', 'n'), 'e')
    # o = ('until', ('not', 'n'), 'g')
    # t1 = ('then', c, o)
    # t2 = ('then', m, o)
    # t3 = ('then', B, o)

    param = WaterWorldParams()
    water_env = WaterWorld(param)
    a = ('eventually', 'a')
    b = ('eventually', 'b')
    c = ('eventually', 'c')
    d = ('eventually', 'd')
    e = ('eventually', 'e')
    f = ('eventually', 'f')
    t1 = ('then', a, b)
    t2 = ('then', c, e)
    t3 = ('then', f, d)
    from src.dfa import _get_truth_assignments

    propositions = ['a', 'b', 'c', 'd', 'e', 'f']
    label_set = _get_truth_assignments(propositions)
    rm = RewardMachine(water_env,
                       DFA(ltl_formula='True',
                           propositions=propositions,
                           label_set=label_set),
                       use_rs=True)
    rm.expand(('or', a, b))
    rm.expand(('or', b, a))
    rm.expand(t3)
    rm.expand(('and', t1, t2))
    a = 1
