import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from _my_reward_machine import *
from _my_network import device

class DQNParams():
    def __init__(self,MEMORY_CAPACITY,Q_NETWORK_ITERATION,BATCH_SIZE,LR,EPSILON,GAMMA):
        self.MEMORY_CAPACITY=MEMORY_CAPACITY
        self.Q_NETWORK_ITERATION=Q_NETWORK_ITERATION
        self.BATCH_SIZE=BATCH_SIZE
        self.LR=LR
        self.EPSILON=EPSILON
        self.GAMMA=GAMMA


class MyDQRM:  # reward machine + learning algorithms
    # continuous states, discrete actions
    # in DQRM, params: learning rate, gamma, epsilon are in reward machines
    def __init__(self,env,rm_list,dqn_params):
        self.env=env
        self.reward_machines=rm_list # list of reward machines
        self.params=dqn_params
        self.state_num = self.env.get_features().size  # state_num of the environment (not the RM)
        self.action_num = len(self.env.get_actions())
        self.rm_num = len(self.reward_machines)
        self.plot_reward = []
        self.learn_step_counter = 0

        self.memory_counter = 0

        self.memory = np.zeros((self.params.MEMORY_CAPACITY, self.state_num * 2 + 2))
        self.label_memory= {}
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward, next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer={}
        for rm_id in range(self.rm_num):
            self.optimizer[rm_id]=[]
            current_rm=self.reward_machines[rm_id]
            for u_i in current_rm.Q:
                self.optimizer[rm_id].append(
                    torch.optim.Adam(current_rm.Q[u_i].eval_net.parameters(), lr=self.params.LR))
        self.loss_func = nn.MSELoss()

    def initialize(self, transfer_method=None):
        self.plot_reward = []
        for rm in self.reward_machines:
            rm.initialize_Q(transfer_method)

    def get_action_epsilon_greedy(self,rm_id, u, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)  # get a 1D array
        if np.random.uniform(0,1,1) <= self.params.EPSILON:  # random policy
            action = np.random.randint(0, self.action_num)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # greedy policy
            action_value = self.reward_machines[rm_id].Q[u].eval_net(state)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state, label):
        # u, u_ is the u-state, next u-state of reward machine
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.params.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.label_memory[index]=label
        self.memory_counter += 1

    def learn(self):
        for rm_id in range(self.rm_num):
            rm=self.reward_machines[rm_id]
            if self.learn_step_counter % self.params.Q_NETWORK_ITERATION == 0:
                for u in rm.Q:
                    rm.Q[u].target_net.load_state_dict(rm.Q[u].eval_net.state_dict())

            # sample batch from memory
            sample_index = np.random.choice(self.params.MEMORY_CAPACITY, self.params.BATCH_SIZE)
            batch_memory = self.memory[sample_index, :]
            # batch_label = []
            # for i in range(self.params.BATCH_SIZE):
            #     batch_label.append(self.label_memory[sample_index[i]])
            batch_state = torch.FloatTensor(batch_memory[:, :self.state_num]).to(device)
            batch_action = torch.LongTensor(
                batch_memory[:, self.state_num:self.state_num + 1].astype(int)).to(device)
            # DQRM does not use reward from memory
            batch_reward = torch.FloatTensor(batch_memory[:,self.state_num+1:self.state_num + 2]).to(device)
            batch_next_state = torch.FloatTensor(batch_memory[:, -self.state_num:]).to(device)
            for u in rm.Q:
                q_eval = rm.Q[u].eval_net(batch_state).gather(1, batch_action)
                ############# test only #########
                q_next = rm.Q[u].target_net(batch_next_state).detach()
                q_target = batch_reward + self.params.GAMMA * q_next.max(1)[0].view(self.params.BATCH_SIZE, 1)
                ####################### low efficiency ############
                # q_next = torch.zeros(self.params.BATCH_SIZE,self.action_num)
                # # DQRM use reward from reward machines
                # batch_reward = torch.zeros(self.params.BATCH_SIZE,1)
                # # time_test = time.time()
                # for i in range(self.params.BATCH_SIZE):
                #     u_ = rm.get_next_state(u, batch_label[i])
                #     batch_reward[i]=rm.get_reward(u,u_,is_training=True)
                #     if rm.is_terminal_state(u_):
                #         q_next[i,:]=torch.zeros(rm.action_num)
                #     else:
                #         q_next[i,:]=rm.Q[u_].target_net(batch_next_state[i,:]).detach()
                # q_target = batch_reward + self.params.GAMMA * q_next.max(1)[0].view(self.params.BATCH_SIZE, 1)
                # print("batch time",time.time() - time_test)
                #####################################################
                # time_test = time.time()
                loss = self.loss_func(q_eval, q_target)
                self.optimizer[rm_id][u].zero_grad()
                loss.backward()
                self.optimizer[rm_id][u].step()
                # print("optimize time",time.time() - time_test)
        self.learn_step_counter += 1

    def run(self, steps_num, max_episode_length):
        current_task = 0
        total_steps = 0
        start = time.time()
        while total_steps<=steps_num:
            current_rm = self.reward_machines[current_task]
            self.env.initialize()
            state = self.env.get_features()
            u = current_rm.get_initial_state()
            for t in range(max_episode_length):  # run for each episode
                if current_rm.is_terminal_state(u) or u==-1:
                    break
                action = self.get_action_epsilon_greedy(current_task,u,state)
                self.env.execute_action(action)  # it will change the state
                state_ = self.env.get_features()
                event = self.env.get_true_propositions()
                u_ = current_rm.get_next_state(u, event)
                r = current_rm.get_reward(u,u_,state,action,state_,is_training=False)
                ############ test of dense reward ##########
                dist=np.exp(-(state[4]**2+state[5]**2))
                ############################################
                self.store_transition(state, action, dist, state_, event)
                # if r==1:
                #     for i in range(10): self.store_transition(state,action,r,state_,event)
                if self.memory_counter>=self.params.MEMORY_CAPACITY:
                    if total_steps % 1==0:
                        self.learn()
                # state transition of both environment and current reward machine
                state, u = state_, u_
                if total_steps==0:
                    self.plot_reward.append(r)
                else:
                    last=self.plot_reward[-1]
                    self.plot_reward.append(last+r)
                total_steps += 1
                if total_steps % 2000==0:
                    print('total_steps:',total_steps)
                    print('ave_reward:',(self.plot_reward[-1]-self.plot_reward[-2000])/2000)
                    print('run_time:',time.time()-start)
                    start=time.time()
            current_task = (current_task+1) % self.rm_num  # go to next task