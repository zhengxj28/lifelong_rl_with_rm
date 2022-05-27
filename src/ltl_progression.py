# from sympy import *
# from sympy.logic import simplify_logic
# from sympy.logic.boolalg import And, Or, Not
import time, collections

"""
This module contains functions to progress co-safe LTL formulas such as:
    (
        'and',
        ('until','True', ('and', 'd', ('until','True','c'))),
        ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
    )
The main function is 'get_dfa' which receives a co-safe LTL formula and progresses 
it over all possible valuations of the propositions_list. It returns all possible progressions 
of the formula in the form of a DFA.
"""

def extract_propositions(ltl_formula):
    return list(set(_get_propositions(ltl_formula)))

def get_dfa(ltl_formula):
    propositions = extract_propositions(ltl_formula)  # type(propostions)=list
    propositions.sort()
    label_set = _get_truth_assignments(propositions)  # power set

    # Creating DFA using progression
    # ltl2state = {'False': -1, ltl_formula: 0}
    ltl2state = {ltl_formula: 0}
    transitions = {}
    terminal_states=set([])
    have_False = False

    visited, queue = set([ltl_formula]), collections.deque([ltl_formula])
    while queue:
        formula = queue.popleft()
        u = ltl2state[formula]
        if formula in ['True', 'False']:
            if formula=='False': have_False=True
            terminal_states.add(ltl2state[formula])
            continue  # terminal states do not need transition
        transitions[u] = {}
        for label in label_set:
            # progressing formula, add transition
            f_progressed = _progress(formula, label)
            if f_progressed not in ltl2state:  # add index for new state
                if f_progressed=='False':
                    ltl2state[f_progressed]=-1
                    have_False=True
                else:
                    ltl2state[f_progressed] = len(ltl2state)-have_False
            # adding edge (transition)
            u_ = ltl2state[f_progressed]
            transitions[u][label]=u_

            if f_progressed not in visited:
                visited.add(f_progressed)
                queue.append(f_progressed)

    # Adding initial and accepting states
    initial_state = 0
    # terminal_states=list(terminal_states)

    return initial_state, terminal_states, ltl2state, transitions, propositions


def _get_truth_assignments(propositions_list):
    # computing all possible value assignments for propositions_list
    truth_assignments = []
    for p in range(2 ** len(propositions_list)):
        truth_assignment = ""
        p_id = 0
        while p > 0:
            if p % 2 == 1:
                truth_assignment += propositions_list[p_id]
            p //= 2
            p_id += 1
        truth_assignments.append(truth_assignment)
    return truth_assignments


def _get_propositions(ltl_formula):
    if type(ltl_formula) == str:
        if ltl_formula in ['True', 'False']:
            return []
        return [ltl_formula]

    if ltl_formula[0] in ['not', 'next', 'eventually']: # unary operator
        return _get_propositions(ltl_formula[1])

    # 'and', 'or', 'until'
    return _get_propositions(ltl_formula[1]) + _get_propositions(ltl_formula[2])


def _is_prop_formula(f):
    # returns True if the formula does not contains temporal operators
    string=str(f)
    return ('next' not in string) & ('until' not in string) & ('eventually' not in string) & ('then' not in string)


def _subsume_until(f1, f2):
    # if f2=('until',..., ...f1...) , i.e. f2->f1 then return True
    if str(f1) not in str(f2):
        return False
    while type(f2) != str:
        if f1 == f2:
            return True
        if f2[0] == 'until':
            f2 = f2[2]
        elif f2[0] == 'and':
            if _is_prop_formula(f2[1]) and not _is_prop_formula(f2[2]):
                f2 = f2[2]
            elif not _is_prop_formula(f2[1]) and _is_prop_formula(f2[2]):
                f2 = f2[1]
            else:
                return False
        else:
            return False
    return False


def _subsume_then(f1, f2):  # test, need to be refined
    # if f2=('then', f1, ...) or ('then',..., f1), i.e. f2->f1 then return True
    if str(f1) not in str(f2):
        return False
    while type(f2) != str:
        if f1 == f2:
            return True
        if f2[0] == 'then':
            if str(f1) in str(f2[1]): return True
            if str(f1) in str(f2[2]): return True
        elif f2[0] == 'and':
            if _is_prop_formula(f2[1]) and not _is_prop_formula(f2[2]):
                f2 = f2[2]
            elif not _is_prop_formula(f2[1]) and _is_prop_formula(f2[2]):
                f2 = f2[1]
            else:
                return False
        else:
            return False
    return False


def _progress(ltl_formula, truth_assignment):
    if type(ltl_formula) == str:
        # True, False, or proposition
        if len(ltl_formula) == 1:
            # ltl_formula is a proposition
            if ltl_formula in truth_assignment:
                return 'True'
            else:
                return 'False'
        return ltl_formula

    if ltl_formula[0] == 'not':
        # negations should be over propositions_list only according to the cosafe ltl syntactic restriction
        result = _progress(ltl_formula[1], truth_assignment)
        if result == 'True':
            return 'False'
        elif result == 'False':
            return 'True'
        else:
            # raise NotImplementedError(
            #     "The following formula doesn't follow the cosafe syntactic restriction: " + str(ltl_formula))
            return ('not',result)

    if ltl_formula[0] == 'and':
        res1 = _progress(ltl_formula[1], truth_assignment)
        res2 = _progress(ltl_formula[2], truth_assignment)
        return _and_ltl(res1,res2)

    if ltl_formula[0] == 'or':
        res1 = _progress(ltl_formula[1], truth_assignment)
        res2 = _progress(ltl_formula[2], truth_assignment)
        return _or_ltl(res1,res2)

    if ltl_formula[0] == 'next':
        return ltl_formula[1]

    if ltl_formula[0] == 'until':
        res1 = _progress(ltl_formula[1], truth_assignment)
        res2 = _progress(ltl_formula[2], truth_assignment)
        if res1 == 'False':
            f1 = 'False'
        elif res1 == 'True':
            f1 = ('until', ltl_formula[1], ltl_formula[2])
        else:
            # f1 = ('and', res1, ('until', ltl_formula[1], ltl_formula[2]))
            f1 = _and_ltl(res1,ltl_formula)
        if res2 == 'True':
            return 'True'
        elif res2 == 'False':
            return f1
        else:
            return res2  # in this case f1 implies res2, thus ('or', res2, f1)==res2

    if ltl_formula[0] == 'eventually':
        res1 = _progress(ltl_formula[1], truth_assignment)
        if res1 == 'True':
            return 'True'
        elif res1 == 'False':
            return ('eventually',ltl_formula[1])
        else:
            return res1  # in this case ltl_formula implies res1, thus ('or', res1, ltl_formula)==res1

    if ltl_formula[0] == 'then':
        if ltl_formula[1]=='True':
            return _progress(ltl_formula[2], truth_assignment)
        elif ltl_formula[1]=='False':
            return 'False'
        else:
            res1 = _progress(ltl_formula[1], truth_assignment)
            if res1=='True':
                return ltl_formula[2]
            elif res1=='False':
                return 'False'
            else:
                return ('then',res1, ltl_formula[2])

def _and_ltl(res1,res2):
    if res1 == 'True' and res2 == 'True': return 'True'
    if res1 == 'False' or res2 == 'False': return 'False'
    if res1 == 'True': return res2
    if res2 == 'True': return res1
    if res1 == res2:   return res1
    if _subsume_until(res1, res2): return res2
    if _subsume_until(res2, res1): return res1
    if _subsume_then(res1, res2): return res2
    if _subsume_then(res2, res1): return res1
    return ('and', res1, res2)

def _or_ltl(res1,res2):
    if res1 == 'True' or res2 == 'True': return 'True'
    if res1 == 'False' and res2 == 'False': return 'False'
    if res1 == 'False': return res2
    if res2 == 'False': return res1
    if res1 == res2:    return res1
    if _subsume_until(res1, res2): return res1
    if _subsume_until(res2, res1): return res2
    if _subsume_then(res1, res2): return res1
    if _subsume_then(res2, res1): return res2
    return ('or', res1, res2)

def _then_ltl(res1,res2):
    if res1=='True': return res2
    if res1=='False': return 'False'
    if res2=='True': return res1
    if res2=='False': return 'False'
    return ('then', res1, res2)

# def _get_formula(truth_assignments, propositions_list):
#     dnfs = []
#     props = dict([(p, symbols(p)) for p in propositions_list])
#     for truth_assignment in truth_assignments:
#         dnf = []
#         for p in props:
#             if p in truth_assignment:
#                 dnf.append(props[p])
#             else:
#                 dnf.append(Not(props[p]))
#         dnfs.append(And(*dnf))
#     formula = Or(*dnfs)
#     formula = simplify_logic(formula, form='dnf')
#     formula = str(formula).replace('(', '').replace(')', '').replace('~', '!').replace(' ', '')
#     return formula

# res1=('then',('until',('not','n'),'c'),('until',('not','n'),'o'))
# res2=('until',('not','n'),'o')
# _and_ltl(res1,res2)