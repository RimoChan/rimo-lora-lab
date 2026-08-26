import random as 原版random
from collections import Counter

_d = {}


class _R:
    def __init__(self, n=4):
        self.buffer = []
        self.n = n

    def random(self):
        if not self.buffer:
            r = 原版random.random()
            for i in range(self.n):
                self.buffer.append((r+i/self.n) % 1)
            原版random.shuffle(self.buffer)
        return self.buffer.pop()


def random(k):
    if k not in _d:
        _d[k] = _R()
    return _d[k].random()


if __name__ == '__main__':
    原版random.seed(1)
    import numpy as np
    a = np.array([random('1') for _ in range(100000)])
    b = np.array([原版random.random() for _ in range(100000)])
    print(0.99 < a.mean()/b.mean() < 1.01)
    print(0.99 < a.var()/b.var() < 1.01)
    
    av = bv = 0
    for _ in range(5000):
        a = np.array([*Counter([int(random('2')*10) for _ in range(100)]).values()])
        # a = np.array([*Counter([int(原版random.random()*10) for _ in range(1000)]).values()])
        b = np.array([*Counter([int(原版random.random()*10) for _ in range(100)]).values()])
        av += a.var()
        bv += b.var()
    print(av/5000, bv/5000)
