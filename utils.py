import numpy as np
from copy import deepcopy


def one_hot(n, i):
    v = np.zeros(n)
    v[i] = 1
    return v

# P can be of any dimension, V is one dimensional
def variance(P, V):
    return np.dot(P, V ** 2) - np.dot(P, V) ** 2


def q_value_iteration(c, p, B=1000):
    """
    find optimal Q function for a SSP with costs c and transition kernel p using value iteration method
    :param c: list or numpy array of shape (nb_states, nb_actions) representing costs
    :param p: list or numpy array of shape (nb_states, nb_actions, nb_states) representing transition
    kernel
    :return: approximately optimal_Q which is a numpy array of size (nb_states, nb_actions).
    """
    eps = .000001
    p = np.array(p)
    n, m = c.shape
    q = np.zeros([n, m])
    while q[0].min() < B: # V(s0) cannot be too large
        q_new = c + np.dot(p, np.min(q, axis=1))
        if np.max(np.abs(q_new - q)) <= eps:
            break
        q = deepcopy(q_new)
        #print(q)
    opt_cost = np.min(q_new, axis=1)
    pi = np.argmin(q_new, axis=1)
    return opt_cost, pi, q_new
