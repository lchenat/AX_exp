import numpy as np
from itertools import product

directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

class GWEnv:
    def __init__(self, h, w, p):
        self.h, self.w = h, w
        self.S, self.A = h * w, 5
        self.p = p
        self.s = (0, 0)
        self.d = [None] + directions

    def step(self, action):
        if action == 0:
            self.s = (0, 0)
            return 0
        x, y = self.s
        if np.random.rand() < self.p:
            action = np.random.choice(4) + 1
        nx, ny = x + self.d[action][0], y + self.d[action][1]
        if 0 <= nx < self.h and 0 <= ny < self.w:
            self.s = nx, ny 
        return self.s[0] * self.w + self.s[1]

    def get_MDP(self, X, g):
        h, w = self.h, self.w
        c = np.ones((self.S, self.A))
        c[g, :] = 0
        p = np.zeros((self.S, self.A, self.S))
        p[:, 0, 0] = 1 # reset
        for i, j in product(range(h), range(w)):
            for a, d in enumerate(directions):
                di, dj = i + d[0], j + d[1]
                if not (0 <= di < h and 0 <= dj < w): di, dj = i, j
                p[i * w + j][a + 1][di * w + dj] += 1 - self.p
                p[i * w + j, :, di * w + dj] += self.p / 4 
        for s in range(self.S):
            if s not in X:
                p[s, :, :] = 0
                p[s, :, 0] = 1 # if not in X, all action is reset
        p[g, :, :] = 0
        p[g, :, g] = 1 # goal is absorbing
        return c, p
