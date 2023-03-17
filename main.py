import numpy as np
from utils import one_hot, variance, q_value_iteration
from env import GWEnv
import ipdb


# X is a set, s0 = 0, a0 is RESET
def VISGO(env, L, X, g, eps, n, delta):
    S, A = env.S, env.A
    c1, c2 = 3, 64
    N = n.sum(axis=-1)
    Np = np.maximum(1, N)
    iota = np.log(2 * len(X) * A * np.maximum(1, N) / delta)
    bP = n / np.expand_dims(Np, -1)
    tP = np.expand_dims(N / (N + 1), -1) * bP + one_hot(S, g)[None,None,:] / np.expand_dims(N + 1, -1)
    V, nV = np.zeros(S), np.full(S, 2 * eps)
    while (nV - V).max() > eps:
        V = nV
        if V.max() > 2 * L:
            #print('V exploded:', V)
            return np.full(S, np.inf), np.zeros(S)
        b = np.maximum(c1 * np.sqrt(np.maximum(1e-8, variance(bP, V) * iota / Np)), c2 * L * iota / Np)
        nQ = np.maximum(0, 1 + np.dot(tP, V) - b)
        nV = nQ.min(axis=-1)
        for s in range(S):
            if s not in X:
                nV[s] = 1 + V[0]
        nV[g] = 0
    pi = np.zeros(S).astype(int)
    pi[list(X)] = nQ[list(X)].argmin(axis=-1)
    return nV, pi


def EXPLORE(env, X, Pi, n, threshold):
    S, A = env.S, env.A
    Snext = set()
    for x in X:
        for a in range(A):
            while n[x][a].sum() < threshold:
                s = env.step(0) # reset
                pi = Pi[x]
                while s != x:
                    s = env.step(pi[s])
                nx = env.step(a)
                n[x, a, nx] += 1
                if nx not in X:
                    Snext.add(nx)
    return n, Snext


def LASD(env, L, eps, delta):
    S, A = env.S, env.A
    K, U, Kp, Pi, N = set(), set(), {0}, {0: np.zeros(S)}, np.zeros((S, A, S))
    calN = {2 ** i for i in range(33)}
    for r in range(1, 2 ** 33):
        eps_VI = 1e-6
        v, gstar, pi = np.inf, None, None
        for g in U:
            V, _pi = VISGO(env, L, K, g, eps_VI, N, 1e-6)
            if V[0] < v:
                v = V[0]
                gstar, pi = g, _pi
        if v > L:
            if not Kp:
                return K, Pi
            K = K.union(Kp)
            Kp, U = set(), set()
            _, U = EXPLORE(env, K, Pi, np.zeros((S, A, S)), 2 * L * np.log(4 * S * A * L * r ** 2 / delta))
            nmin =  int(10 * L ** 2 * len(K) * np.log(S * r / delta))
            N, _ = EXPLORE(env, K, Pi, N, nmin)
            print('update K:', r, v, K, U, nmin)
        else: # policy evaluation
            #print(r, gstar, pi)
            tau, lambd = 0, int(20 * np.log(L * r / (eps * delta)) / eps ** 2)
            skip = False
            for j in range(lambd):
                s = env.step(0) # reset
                while s != gstar:
                    a = pi[s]
                    ns = env.step(a)
                    N[s, a, ns] += 1
                    if int(N.sum()) in calN or (s in K and int(N[s, a].sum()) in calN): # skip
                        skip = True
                        break
                    tau += 1
                    s = ns
                if skip or tau / lambd > v + eps * L / 2: # failure
                    skip = True
                    break
            if not skip: # success
                Kp.add(gstar)
                U.remove(gstar)
                Pi[gstar] = pi


def compute_K(env, L):
    S, B = env.S, 2 * L
    Ks = [[0]]
    while True:
        K, nK = sum(Ks, []), []
        for g in range(S):
            if g in K: continue
            c, P = env.get_MDP(K, g)
            V, pi, Q = q_value_iteration(c, P, B)
            if V[0] <= L:
                nK.append(g)
        print(K, nK)
        if not nK: break
        Ks.append(nK)
    return Ks


if __name__ == "__main__":
    L, eps, delta = 4, 1e-2, 1e-3
    env = GWEnv(4, 4, 0.2)

    K, Pi = LASD(env, L, eps, delta)
    print(K)
    print(Pi)

    #print(compute_K(env, L))
