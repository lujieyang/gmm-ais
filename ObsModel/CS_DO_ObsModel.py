import numpy as np

class CS_DO_ObsModel:
    def __init__(self, S, O, om):
        self.S = S
        self.O = O
        self.p = om

        no = len(self.O.values)

        # p(s) should be uniform in 's'. We compute p(s) as
        #       p(s) = sum_o p(o,s)
        # and, assuming it is constant, we scale p(o,s) so that
        # p(s) is actually uniform.
        o = self.p[1]
        for i in range(1, no):
            o = o+self.p[i]

        ps = 0
        num_samples = 10
        for i in range(num_samples):
            ps += o.Value(self.S.rand())

        ps /= num_samples
        scale = self.S.UniformProbability()/ps
        for i in range(no):
            self.p[i] = self.p[i]*scale

    def GetObsModelFixedS(self, s):
        """
        Defines p(o|s).
        Instantiates the observation model for a particular observation 'o'.
        We use the fact that p(o|s)=p(o,s)/p(s) and we assume a uniform
        distribution in s.
        :param s:
        :return:
        """
        p = []
        for om_p in self.p:
            p.append(om_p.Value(s))
        p_sum = np.sum(p)
        for i in range(len(p)):
            p[i] = p[i]/p_sum
        return p

    def GetObsModelFixedO(self, o):
        ps = self.S.UniformProbability()
        p = self.p[o]/ps
        return p

    def Update(self, b, o, Sp):
        """
        Belief evolution under an observation model
        :param b:
        :param o:
        :param Sp: the sub-space where beliefs are defined and is used to
        ensure that, after the update, the belief is still inside this
        sub-space.
        :return:
        """
        po = self.GetObsModelFixedO(o)
        bOut = b.Update(po, Sp)
        return bOut






