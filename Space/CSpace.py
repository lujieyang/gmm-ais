import numpy as np

class CSpace:
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.dim = np.size(self.min)
        self.range = self.max - self.min

    def rand(self):
        if self.dim == 1:
            return self.min + np.random.rand() * self.range
        else:
            return self.min + np.random.rand(self.dim)*self.range

    def UniformProbability(self):
        """
        :return: Uniform probability value on a continuous space
        """
        return np.prod(np.ones((1, self.dim))/self.range, axis=0)

    def Crop(self, s):
        """
        Forces a state to be in the continuous sub-space.
        :param s:
        :return:
        """
        try:
            return np.min([self.max, np.max([self.min, s.item()])])
        except:
            return np.min([self.max, np.max([self.min, s])])
