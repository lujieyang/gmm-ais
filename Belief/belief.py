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
        self.gm = gm1
        GMixture.__init__(self, gm=gm1)

    def Prediction(self, t, Sp):
        return GBelief(self.Compose(t).Crop(Sp), self.maxC)

    def Update(self, po, Sp):
        bo = po * self  #.gm
        return GBelief(bo.Crop(Sp), self.maxC)

