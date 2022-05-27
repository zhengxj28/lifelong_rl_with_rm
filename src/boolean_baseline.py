from src._my_reward_machine import RewardMachine
from src.dfa import DFA
from src._my_network import device
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time

use_normalize = True   # we always normalize Q-values during transferring
class LifelongLearning():
    def __init__(self, env, propositions, label_set, gamma, alpha, epsilon, algorithm):
        self.q_network_iteration=1
        self.env = env
        self.plot_reward = []
        self.plot_steps = []  # steps to complete the task
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.state_num = self.env.get_features().size  # state_num of the environment (not the RM)
        self.action_num = len(self.env.get_actions())
        self.memory_rm = RewardMachine(env,
                                       dfa=DFA(ltl_formula='True', propositions=propositions, label_set=label_set),
                                       transfer=None, use_rs="rs" in algorithm)
        # algorithm="TQRM" means QRM with knowledge transfer, i.e. LSRM
        if algorithm in ["QRM", "QRMrs", "equiv", "TQRM", "advisor", "TQRM_advisor","TQRMrs",
                         "TQRMaverage", "TQRMmax", "TQRMleft", "TQRMright"]:
            self.algorithm = algorithm  # algorithm is "QRM","equiv","TQRM","advisor","TQRM_advisor"
        else:  # default algorithm
            self.algorithm = "equiv"
            print("Inputted algorithm error. Default: equiv.")
        self.advisor_Q = self.memory_rm.build_Q()
        self.task_num = 0  # number of learned tasks

        if not self.env.is_discrete:  # continuous cases
            ####### DQN parameters ########
            self.buffer_capacity=5000
            self.learn_step_counter = 0
            self.buffer_counter = 0
            self.batch_size=32
            ################################
            self.buffer = np.zeros((self.buffer_capacity, self.state_num * 2 + 2))
            self.u_buffer=np.zeros((self.buffer_capacity, self.memory_rm.max_states),dtype=int)
            self.r_buffer=np.zeros((self.buffer_capacity, self.memory_rm.max_states))
            self.optimizer = {}
            self.loss_func = nn.MSELoss()
            for u, dqn in self.memory_rm.Q.items():
                # type(dqn)=_my_network.DQN
                self.optimizer[u] = torch.optim.Adam(dqn.eval_net.parameters(), lr=self.alpha)
                self.loss_func = nn.MSELoss()


    def update_memory(self, new_formula, new_states):  # update the memory RM, transfer Q-functions
        new_states = self.memory_rm.expand(new_formula)+new_states  # Q-functions of rm have been extended
        for u in new_states:  # extend advisor, optimizer
            self.advisor_Q[u] = self.memory_rm.build_one_Q()
            if not self.env.is_discrete:
                dqn=self.memory_rm.Q[u]
                self.optimizer[u] = torch.optim.Adam(dqn.eval_net.parameters(), lr=self.alpha)
        if self.algorithm in ["QRM","QRMrs"]:
            self.memory_rm.initialize_Q()  # learn from scratch
        if "TQRM" in self.algorithm:
            for u in new_states:
                self.memory_rm.Q[u].eval = self.non_equiv_transfer(u, new_states)
                self.memory_rm.Q[u].tar=self.memory_rm.Q[u].eval.copy()
        if "advisor" in self.algorithm:
            for u in new_states:
                self.advisor_Q[u].eval = self.non_equiv_transfer(u, new_states)
        return new_states

    def get_action_epsilon_greedy(self, u, state):
        # u is a state of rm
        if self.env.is_discrete:
            if np.random.uniform() < self.epsilon:  # choose randomly
                if "advisor" in self.algorithm:
                    action = int(self.advisor_Q[u][state].argmax())
                else:
                    action = np.random.randint(0, self.action_num)
            else:
                action = int(self.memory_rm.Q[u].eval[state].argmax())
            return action
        else:
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)  # get a 1D array
            if np.random.uniform() < self.epsilon:  # random policy
                # if "advisor" in self.algorithm:
                action = np.random.randint(0, self.action_num)
                # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            else:  # greedy policy
                action_value = self.memory_rm.Q[u].eval_net(state)
                action = torch.max(action_value, 1)[1].data.cpu().numpy()
                # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            return action

    def non_equiv_transfer(self, u, new_states): # return np.array
        if u not in new_states:
            Q =self.memory_rm.Q[u].eval.copy()
            if use_normalize:
                from sklearn.preprocessing import scale
                Q =scale(Q,axis=1,with_mean=True,with_std=False)+self.memory_rm.use_rs  # normalize
            return Q

        formula = self.memory_rm.dfa.state2ltl[u]
        ########### TQRM ###########################
        if formula[0] in ['and', 'or','then']:
            u1 = self.memory_rm.dfa.ltl2state[formula[1]]  # formula[1] in dfa?
            u2 = self.memory_rm.dfa.ltl2state[formula[2]]
            Q1 = self.non_equiv_transfer(u1, new_states)
            Q2 = self.non_equiv_transfer(u2, new_states)
            ########### for Experiment 1,2 only ################
            if self.algorithm=="TQRMaverage":
                return (Q1+Q2)/2
            elif self.algorithm=="TQRMmax":
                Q = np.zeros([self.state_num, self.action_num])
                Q1_max = np.max(Q1, axis=1)
                Q2_max = np.max(Q2, axis=1)
                for s in range(self.state_num):
                    if Q1_max[s] > Q2_max[s]:
                        Q[s, :] = Q1[s, :]
                    else:
                        Q[s, :] = Q2[s, :]
                return Q
            elif self.algorithm=="TQRMleft": return Q1
            elif self.algorithm=="TQRMright": return Q2
            ################# TQRM ##########################
            if formula[0] == 'and':
                return (Q1 + Q2) / 2
            elif formula[0] == 'or':
                Q = np.zeros([self.state_num, self.action_num])
                Q1_max = np.max(Q1, axis=1)
                Q2_max = np.max(Q2, axis=1)
                for s in range(self.state_num):
                    if Q1_max[s] > Q2_max[s]:
                        Q[s, :] = Q1[s, :]
                    else:
                        Q[s, :] = Q2[s, :]
                return Q
            elif formula[0] == 'then':
                return Q1
        else:
            return self.memory_rm.build_one_Q().eval
        #########################################################

    def non_equiv_transfer_deep(self, u, new_states):
        return self.memory_rm.build_one_Q()

    def learn(self, state, action, state_, event):
        for u in self.memory_rm.dfa.state2ltl:
            if self.memory_rm.is_terminal_state(u): continue
            u_ = self.memory_rm.get_next_state(u, event)
            r = self.memory_rm.get_reward(u, u_, state, action, state_, is_training=True)
            if self.memory_rm.is_terminal_state(u_):
                self.memory_rm.Q[u].eval[state, action] = r
            else:  # use Q.tar to update Q.eval
                # here we do not use Q.tar
                self.memory_rm.Q[u].eval[state, action] += \
                    self.alpha * (
                            r + self.gamma * self.memory_rm.Q[u_].eval[state_, :].max() -
                            self.memory_rm.Q[u].eval[state, action]
                    )
                # self.memory_rm.Q[u].eval[state, action] += \
                #     self.alpha * (
                #             r + self.gamma * self.memory_rm.Q[u_].tar[state_, :].max() -
                #             self.memory_rm.Q[u].eval[state, action]
                #     )

    def update_buffer(self, state, action, reward, next_state, label):
        # u, u_ is the u-state, next u-state of reward machine
        transition = np.hstack((state, [action, reward], next_state))
        index = self.buffer_counter % self.buffer_capacity
        self.buffer[index, :] = transition
        try:
            self.u_buffer[index,:]=self.memory_rm.LU_table[label]
            self.r_buffer[index,:]=self.memory_rm.LR_table[label]
        except KeyError as e:
            label=e.args[0][0]
            self.u_buffer[index, :] = self.memory_rm.LU_table[label]
            self.r_buffer[index, :] = self.memory_rm.LR_table[label]
        self.buffer_counter += 1

    def learn_deep(self):
        if self.learn_step_counter % self.q_network_iteration == 0:
            for u in self.memory_rm.Q:
                self.memory_rm.Q[u].target_net.load_state_dict(self.memory_rm.Q[u].eval_net.state_dict())

        # sample batch from memory
        sample_index = np.random.choice(self.buffer_capacity, self.batch_size)
        batch_memory = self.buffer[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.state_num]).to(device)
        batch_action = torch.LongTensor(
            batch_memory[:, self.state_num:self.state_num + 1].astype(int)).to(device)
        # DQRM does not use reward from memory
        # batch_reward = torch.FloatTensor(batch_memory[:, self.state_num + 1:self.state_num + 2]).to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.state_num:]).to(device)
        batch_all_next_u = torch.LongTensor(self.u_buffer[sample_index,:]).to(device)
        batch_all_reward = torch.FloatTensor(self.r_buffer[sample_index,:]).to(device)
        for u,dqn in self.memory_rm.Q.items():
            q_eval = dqn.eval_net(batch_state).gather(1, batch_action)
            ############# test only #########
            # q_next = dqn.target_net(batch_next_state).detach()
            # q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            q_all_next_u = torch.zeros((self.memory_rm.max_states,self.batch_size,self.action_num)).to(device)
            for u_,dqn_ in self.memory_rm.Q.items():
                q_all_next_u[u_,:,:]=dqn_.target_net(batch_next_state).detach().to(device)
            batch_next_u = batch_all_next_u[:,u]
            q_next = q_all_next_u.gather(0,batch_next_u.unsqueeze(0).unsqueeze(2).repeat(1,1,self.action_num)).squeeze(0)
            batch_reward = batch_all_reward[:,u].unsqueeze(1)
            q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            loss = self.loss_func(q_eval, q_target)
            self.optimizer[u].zero_grad()
            loss.backward()
            self.optimizer[u].step()
        self.learn_step_counter += 1

    def qrm_run(self, steps_num, max_episode_length, phase_tasks):
        start_time = time.time()
        total_steps = 0
        self.task_num += len(phase_tasks)
        if ("advisor" in self.algorithm): self.epsilon = 0.9
        while True:
            for formula in phase_tasks:  # run through each task like qrm
                self.env.initialize(random_init=True)
                state = self.env.get_state()  # get_state() function was redefined
                u = self.memory_rm.dfa.ltl2state[formula]  # initial state
                episode_steps = 0
                while episode_steps < max_episode_length:  # run for each episode
                    # if total_steps % self.q_network_iteration==0:
                    #     for each_u in self.memory_rm.Q:
                    #         if self.env.is_discrete: self.memory_rm.Q[each_u].tar=self.memory_rm.Q[each_u].eval.copy()
                    #         else: self.memory_rm.Q[each_u].target_net.load_state_dict(self.memory_rm.Q[each_u].eval_net.state_dict())
                    if total_steps >= steps_num: break
                    if self.memory_rm.is_terminal_state(u) or u==-1:
                        break
                    action = self.get_action_epsilon_greedy(u, state)
                    self.env.execute_action(action)  # it will change the state
                    state_ = self.env.get_state()
                    event = self.env.get_true_propositions()
                    u_ = self.memory_rm.get_next_state(u, event)
                    r = self.memory_rm.get_reward(u, u_, state, action, state_, is_training=False)
                    if self.env.is_discrete: self.learn(state, action, state_, event)
                    else:
                        ######## store (s,a,r,s') in the buffer for continuous cases ###########
                        self.update_buffer(state, action, r, state_, event)
                        if total_steps>=self.buffer_capacity: self.learn_deep()
                    # state transition of both environment and current reward machine
                    state, u = state_, u_
                    ######## store the result, rewards and steps to complete task ###########
                    r=1 if r>=1 else 0
                    if total_steps == 0:
                        self.plot_reward.append(r)
                    else:
                        last = self.plot_reward[-1]
                        self.plot_reward.append(last + r)
                    if total_steps < max_episode_length:
                        self.plot_steps.append(max_episode_length)
                    elif (self.plot_reward[-1]-self.plot_reward[-max_episode_length])!=0:
                        average_steps=max_episode_length/(self.plot_reward[-1]-self.plot_reward[-max_episode_length])
                        self.plot_steps.append(average_steps)
                    else: self.plot_steps.append(max_episode_length)
                    ################################################################
                    episode_steps += 1
                    total_steps += 1
                    if not self.env.is_discrete and total_steps%10000==0:
                        print("Total steps:",total_steps,". Steps to complete: ",self.plot_steps[-1])
                        print("Runtime per 10000 steps:"+str(time.time()-start_time))
                        start_time=time.time()
                ######## an episode end ############
                if ("advisor" in self.algorithm) and self.epsilon>0.1: self.epsilon *= 0.95
            if total_steps >= steps_num: break
        if "advisor" in self.algorithm:
            for u in self.memory_rm.Q:
                self.advisor_Q[u].eval = self.memory_rm.Q[u].eval.copy()
                self.advisor_Q[u].tar = self.memory_rm.Q[u].tar.copy()

def run_lifelong(tasks,
                 steps_num,
                 repeated_test_times,
                 env,
                 propositions,
                 label_set,
                 gamma,
                 alpha,
                 epsilon,
                 max_episode_length,
                 algorithm,
                 save_data=True,
                 data_name="",
                 directory="data2/"):
    if type(tasks[0])!=list:  # if tasks=[task1,task2,...], then convert to [[task1],[task2],...]
        temp_tasks=[]
        for formula in tasks:
            temp_tasks.append([formula])
        tasks=temp_tasks
    plot_result = np.zeros([len(tasks), repeated_test_times, steps_num])
    for t in range(repeated_test_times):  # lifelong learning
        np.random.seed(t)
        lifelong_model = LifelongLearning(env,
                                          propositions,
                                          label_set,
                                          gamma, alpha, epsilon, algorithm)
        start=time.time()
        for task_i in range(len(tasks)):
            new_states = []
            phase_tasks = tasks[task_i]  # type(phase_tasks)=list
            for formula in phase_tasks:
                new_states=lifelong_model.update_memory(formula,new_states)
            lifelong_model.qrm_run(steps_num, max_episode_length, phase_tasks=phase_tasks)
            ############ store the reward ###############
            plot_result[task_i, t, :] = np.array(lifelong_model.plot_reward)
            ########### store the steps #################
            # plot_result[task_i, t, :] = np.array(lifelong_model.plot_steps)
            lifelong_model.plot_reward = []
            lifelong_model.plot_steps = []
        print("Test times: ", t, algorithm, "Run time:",time.time()-start)
        # plot_ave=np.average(plot_result,axis=1)
        # plot_std=np.std(plot_result,axis=1)
    if use_normalize: data_name="data2/"+data_name + algorithm + "norm.npy"
    else: data_name=directory+data_name + algorithm + ".npy"
    if save_data:
        np.save(data_name, plot_result)
    return plot_result

