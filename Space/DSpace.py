import numpy as np

class DSpace:
    def __init__(self, max):
        self.max = max
        self.values = np.arange(self.max)

    def rand(self):
        return np.ceil(np.random.rand()*self.max)
