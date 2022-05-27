import src.ltl_progression
import collections
from src.ltl_progression import _get_truth_assignments,_and_ltl,_or_ltl,_then_ltl,_progress

class DFA:
    def __init__(self, ltl_formula,**kwargs):
        # Progressing formula
        if len(kwargs)<=2:
            initial_state, terminal, ltl2state, transitions, propositions = src.ltl_progression.get_dfa(ltl_formula)
            if 'propositions' in kwargs:
                propositions = kwargs['propositions']
        else:
            ltl2state=kwargs['ltl2state']
            initial_state=kwargs['initial_state']
            propositions=kwargs['propositions']
            transitions=kwargs['transitions']
            terminal=kwargs['terminal']

        # setting the DFA
        self.formula = ltl_formula
        self.ltl2state = ltl2state  # dictionary from ltl to state
        self.state = initial_state  # current state <- initial state id
        self.propositions = set(propositions)
        # type(label_set)=list
        if 'label_set' in kwargs:
            self.label_set=kwargs['label_set']
        else:
            self.label_set=_get_truth_assignments(propositions)
        self.transitions = transitions
        self.terminal = terminal  # set of terminal states

        self.state2ltl = dict([[v,k] for k,v in self.ltl2state.items()])
        self.dfa_state_num     = len(ltl2state) # num of all dfa states
        self.non_terminal_state_num=self.dfa_state_num-len(self.terminal)  #num of non_terminal states


    def progress(self, true_props):
        """
        This method progress the DFA and returns False if we fall from it
        """
        self.state = self._get_next_state(self.state, true_props)

    def _get_next_state(self, v1, true_props):  # transition function
        try:
            return self.transitions[v1][true_props]
        except KeyError as e:
            if type(e.args[0])==int and self.state2ltl[e.args[0]] in ['True','False']:
                return v1
            if type(e.args[0])==str:
                # note that true_props from env may include proposition that is not in dfa
                # thus we need to do a intersection
                label = ''
                for p in true_props:
                    if p in self.propositions:
                        label += p
                try:
                    return self.transitions[v1][label]
                except:
                    return -1

        # if true_props in self.transitions[v1]:
        #     return self.transitions[v1][true_props]
        # else:
        #     return -1

    def progress_LTL(self, ltl_formula, true_props):
        """
        Returns the progression of 'ltl_formula' given 'true_props'
        Special cases:
            - returns 'None' if 'ltl_formula' is not part of this DFA
            - returns 'False' if the formula is broken by 'true_props'
            - returns 'True' if the terminal state is reached
        """
        if ltl_formula not in self.ltl2state:
            raise NameError('ltl formula ' + ltl_formula + " is not part of this DFA")
        return self.get_LTL(self._get_next_state(self.ltl2state[ltl_formula], true_props))

    def in_terminal_state(self):
        return self.state in self.terminal

    def get_LTL(self, s = None):
        if s is None: s = self.state
        return self.state2ltl[s]

    def is_game_over(self):
        """
        For now, I'm considering gameover if the agent hits a DFA terminal state or if it falls from it
        """
        return self.in_terminal_state() or self.state == -1

    def __str__(self):
        aux = []
        for v1 in self.transitions:
            aux.extend([str((v1,v2,self.transitions[v1][v2])) for v2 in self.transitions[v1]])
        return "\n".join(aux)

    def compose(self,operation,f1,f2):
        pass

    def expand_dfa(self,formula):
        new_states=[] # new non-terminal states after expansion
        if formula in self.ltl2state:
            pass
        # elif formula[0] in ['and','or','then']:
        #     self.expand(formula[1])
        #     self.expand(formula[2])
        #     self.compose(formula[0],formula[1],formula[2])
        elif formula=='False':
            self.ltl2state[formula]=-1
            # self.state2ltl[-1]=formula
            self.terminal.add(-1)
        elif formula=='True':
            index=len(self.ltl2state) - (-1 in self.terminal)
            self.ltl2state[formula]=index
            # self.state2ltl[index]=formula
            self.terminal.add(index)
        else: # formula is brand new
            queue=collections.deque([formula])
            index=len(self.ltl2state) - (-1 in self.terminal)  # -1 is the index of 'False'
            self.ltl2state[formula] = index
            # self.state2ltl[index]=formula
            new_states.append(index)
            while queue:
                psi=queue.popleft()
                u=self.ltl2state[psi] # u is the index of state (type: int)
                if psi in ['True', 'False']:
                    self.terminal.add(self.ltl2state[psi])
                    continue  # terminal states do not need transition

                ############### extend states of psi_1, where psi=('then',psi_1,psi_2) ###########
                if psi[0]=='then' and psi[1] not in self.ltl2state:
                    queue.append(psi[1])
                    index = len(self.ltl2state) - (-1 in self.terminal)
                    self.ltl2state[psi[1]] = index
                    if psi[1]!='True': new_states.append(index)
                ###############################################################
                self.transitions[u] = {}
                for label in self.label_set:
                    # progressing formula, add transition
                    psi_ = _progress(psi, label)
                    if psi_ not in self.ltl2state:  # add index for new state
                        if psi_ == 'False':
                            self.ltl2state[psi_] = -1
                            self.state2ltl[-1]=psi_
                        else:
                            index=len(self.ltl2state) - (-1 in self.terminal)
                            self.ltl2state[psi_] = index
                            # self.state2ltl[index]=psi_
                            if psi_ != 'True': new_states.append(index)
                        queue.append(psi_)
                    # adding edge (transition)
                    u_ = self.ltl2state[psi_]
                    self.transitions[u][label] = u_

        self.state2ltl = dict([[v, k] for k, v in self.ltl2state.items()])
        return new_states


"""
Evaluates 'formula' assuming 'true_props' are the only true propositions_list and the rest are false. 
e.g. _evaluate_DNF("a&b|!c&d","d") returns True 
"""
def _evaluate_DNF(formula,true_props):
    # ORs
    if "|" in formula:
        for f in formula.split("|"):
            if _evaluate_DNF(f,true_props):
                return True
        return False
    # ANDs
    if "&" in formula:
        for f in formula.split("&"):
            if not _evaluate_DNF(f,true_props):
                return False
        return True
    # NOT
    if formula.startswith("!"):
        return not _evaluate_DNF(formula[1:],true_props)

    # Base cases
    if formula == "True":  return True
    if formula == "False": return False
    return formula in true_props


############ dfa composition #######################
def and_dfa(dfa1,dfa2):
    P1=dfa1.propositions  # set of propositions_list
    P2=dfa2.propositions
    ltl_formula=_and_ltl(dfa1.formula,dfa2.formula)  # target formula
    propositions=P1 | P2  # union of sets
    propositions_list=list(propositions)  # turn set into list
    propositions_list.sort()
    label_set = _get_truth_assignments(propositions_list)  # power set

    # generate states
    have_False = False  # if 'False' in ltl2state then have_False=True
    ltl2state = {ltl_formula:0}
    terminal = set([])  # id of terminal states

    u0_1,u0_2=dfa1.ltl2state[dfa1.formula],dfa2.ltl2state[dfa2.formula]
    visited, queue = set([ltl_formula]), collections.deque([(u0_1,u0_2)])
    transitions={}
    while queue:
        u1,u2 = queue.popleft()
        formula = _and_ltl(dfa1.state2ltl[u1],dfa2.state2ltl[u2])
        u = ltl2state[formula]
        if formula in ['True', 'False']:
            if formula == 'False': have_False = True
            terminal.add(ltl2state[formula])
            continue  # terminal states do not need transition
        transitions[u] = {}
        for label in label_set:  # type(label)=str
            true_props1, true_props2 = '', ''
            for l in label:
                if l in P1: true_props1+=l
                if l in P2: true_props2+=l
            # type of paras: _get_next_state(self, (int) v1, (str) true_props)
            u1_ =dfa1._get_next_state(u1,true_props1)
            u2_ =dfa2._get_next_state(u2,true_props2)
            f_progressed=_and_ltl(dfa1.state2ltl[u1_],dfa2.state2ltl[u2_])
            if f_progressed not in ltl2state:  # add index for new state
                if f_progressed == 'False':
                    ltl2state[f_progressed] = -1
                    have_False = True
                else:
                    ltl2state[f_progressed] = len(ltl2state) - have_False

            # add transitions
            transitions[u][label]=ltl2state[f_progressed]

            if f_progressed not in visited:
                visited.add(f_progressed)
                queue.append((u1_,u2_))
    tar_dfa = DFA(ltl_formula,
                  propositions=propositions,
                  initial_state=ltl2state[ltl_formula],
                  terminal=terminal,
                  ltl2state=ltl2state,
                  transitions=transitions)
    return tar_dfa

def neg_dfa(dfa):
    tar_ltl2state={}
    for ltl in dfa.ltl2state:
        if ltl=='False':
            tar_ltl='True'
        elif ltl=='True':
            tar_ltl='False'
        elif ltl[0] == 'not':
            tar_ltl=ltl[1]
        else:
            tar_ltl=('not',ltl)
        tar_ltl2state[tar_ltl]=dfa.ltl2state[ltl]
    tar_dfa=DFA(dfa.state2ltl[0],
                propositions=dfa.propositions,
                initial_state=dfa.ltl2state[dfa.formula],
                terminal=dfa.terminal,
                ltl2state=tar_ltl2state,
                transitions=dfa.transitions)
    return tar_dfa

def or_dfa(dfa1,dfa2):
    P1=dfa1.propositions  # set of propositions_list
    P2=dfa2.propositions
    ltl_formula=_or_ltl(dfa1.formula,dfa2.formula)  # target formula
    propositions=P1 | P2  # union of sets
    propositions_list=list(propositions)  # turn set into list
    propositions_list.sort()
    label_set = _get_truth_assignments(propositions_list)  # power set

    # generate states
    have_False = False  # if 'False' in ltl2state then have_False=True
    ltl2state = {ltl_formula:0}
    terminal = set([])  # id of terminal states

    u0_1,u0_2=dfa1.ltl2state[dfa1.formula],dfa2.ltl2state[dfa2.formula]
    visited, queue = set([ltl_formula]), collections.deque([(u0_1,u0_2)])
    transitions={}
    while queue:
        u1,u2 = queue.popleft()
        formula = _or_ltl(dfa1.state2ltl[u1],dfa2.state2ltl[u2])
        u = ltl2state[formula]
        if formula in ['True', 'False']:
            if formula == 'False': have_False = True
            terminal.add(ltl2state[formula])
            continue  # terminal states do not need transition
        transitions[u] = {}
        for label in label_set:  # type(label)=str
            true_props1, true_props2 = '', ''
            for l in label:
                if l in P1: true_props1+=l
                if l in P2: true_props2+=l
            # type of paras: _get_next_state(self, (int) v1, (str) true_props)
            u1_ =dfa1._get_next_state(u1,true_props1)
            u2_ =dfa2._get_next_state(u2,true_props2)
            f_progressed=_or_ltl(dfa1.state2ltl[u1_],dfa2.state2ltl[u2_])
            if f_progressed not in ltl2state:  # add index for new state
                if f_progressed == 'False':
                    ltl2state[f_progressed] = -1
                    have_False = True
                else:
                    ltl2state[f_progressed] = len(ltl2state) - have_False

            # add transitions
            transitions[u][label]=ltl2state[f_progressed]

            if f_progressed not in visited:
                visited.add(f_progressed)
                queue.append((u1_,u2_))
    tar_dfa = DFA(ltl_formula,
                  propositions=propositions,
                  initial_state=ltl2state[ltl_formula],
                  terminal=terminal,
                  ltl2state=ltl2state,
                  transitions=transitions)
    return tar_dfa

def then_dfa(dfa1,dfa2):
    P1=dfa1.propositions  # set of propositions_list
    P2=dfa2.propositions
    ltl_formula=_then_ltl(dfa1.formula,dfa2.formula)  # target formula
    propositions=P1 | P2  # union of sets
    propositions_list=list(propositions)  # turn set into list
    propositions_list.sort()
    label_set = _get_truth_assignments(propositions_list)  # power set

    # generate states
    have_False = False  # if 'False' in ltl2state then have_False=True
    ltl2state = {ltl_formula:0}
    terminal = set([])  # id of terminal states

    u0_1, u0_2 = dfa1.ltl2state[dfa1.formula], dfa2.ltl2state[dfa2.formula]
    visited, queue = set([ltl_formula]), collections.deque([(u0_1, u0_2)])
    transitions={}

    while queue:
        u1,u2 = queue.popleft()
        formula = _then_ltl(dfa1.state2ltl[u1], dfa2.state2ltl[u2])
        u = ltl2state[formula]
        if formula in ['True', 'False']:
            if formula == 'False': have_False = True
            terminal.add(ltl2state[formula])
            continue  # terminal states do not need transition
        transitions[u] = {}
        for label in label_set:  # type(label)=str
            true_props1, true_props2 = '', ''
            for l in label:
                if l in P1: true_props1+=l
                if l in P2: true_props2+=l

            # type of paras: _get_next_state(self, (int) v1, (str) true_props)
            if dfa1.state2ltl[u1]=='True':
                u1_=u1
                u2_=dfa2._get_next_state(u2,true_props2)
                f_progressed=dfa2.state2ltl[u2_]
            else:
                u1_=dfa1._get_next_state(u1,true_props1)
                u2_=dfa2.ltl2state[dfa2.formula]
                f_progressed=_then_ltl(dfa1.state2ltl[u1_],dfa2.formula)
            if f_progressed not in ltl2state:  # add index for new state
                if f_progressed == 'False':
                    ltl2state[f_progressed] = -1
                    have_False = True
                else:
                    ltl2state[f_progressed] = len(ltl2state) - have_False

            # add transitions
            transitions[u][label]=ltl2state[f_progressed]
            if f_progressed not in visited:
                visited.add(f_progressed)
                queue.append((u1_,u2_))
    tar_dfa = DFA(ltl_formula,
                  propositions=propositions,
                  initial_state=ltl2state[ltl_formula],
                  terminal=terminal,
                  ltl2state=ltl2state,
                  transitions=transitions)
    return tar_dfa


if __name__=='__main__':
    dfa=DFA('True',propositions=['c','m','o','a','b','n'],label_set=['a','b','c','m','o','n',''])
    dfa.expand_dfa(
        ('then',
         ('then',('eventually','c'),('eventually','o')),
         ('then',('eventually','m'),('eventually','o'))))
    # dfa.expand(('then',('eventually','m'),('eventually','o')))
    dfa.expand_dfa(('and',('then',('eventually','c'),('eventually','o')),('then',('eventually','m'),('eventually','o'))))
    con=('not','n')
    dfa.expand_dfa(('until',con,('then',('and',con,'a'),('and',con,'b'))))
    # t1=('then','a',('then',('eventually','b'),('eventually','d')))
    # t2=('then','a',('then',('eventually','c'),('eventually','d')))
    # dfa=DFA(('eventually',('and',t1,('eventually',t2))))
    # dfa=DFA(('until',('then','a','b' ),'c'))

    # dfa=DFA(('eventually','a'))
    # dfa._get_next_state(0,'b')
    # a=1

