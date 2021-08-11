import numpy as np
from Lib.GMixture import GMixture

class Belief:
    def __init__(self, b=None):
        if b is None:
            self.dummy = 1
        else:
            self = b

class GBelief(Belief, GMixture):
    def __init__(self, gm, ncBelief):

        self.maxC = ncBelief
        gm1 = (gm.Compress(self.maxC)).Normalize()
        GMixture.__init__(self, gm=gm1)

    def Prediction(self, t, Sp):
        return GBelief(self.Compose(t).Crop(Sp), self.maxC)

    def Update(self, po, Sp):
        bo = po * self
        return GBelief(bo.Crop(Sp), self.maxC)

    def Expectation(self, x):
        return self.ProductInt(x)

    def to_array(self):
        g_dim = self.g[0].dim
        input_dim = self.maxC * (1 + g_dim + g_dim ** 2)  # w, mean, flatten(Sigma)
        # If the belief object has mixtures fewer than ncBelief, fill with zeros
        b = np.zeros(input_dim)
        nBelief = len(self.w)
        b[:nBelief] = self.w
        b[self.maxC:self.maxC + nBelief] = [g.m for g in self.g]
        b[self.maxC * (g_dim + 1):self.maxC * (g_dim + 1) + nBelief] = [g.S for g in self.g]
        return b

    def sample_array(self, nb):
        X = np.linspace(-20, 20, nb)
        Y = self.Value(X)
        return Y


