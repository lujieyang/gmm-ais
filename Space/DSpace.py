import numpy as np

class DSpace:
    def __init__(self, max):
        self.max = max
        self.values = np.arange(self.max)

    def rand(self):
        return np.ceil(np.random.rand()*self.max)

    def hard_negative(self):
        return np.where(np.random.multinomial(1, [0.25, 0.25, 0.5]) == 1)[0][0] + 1