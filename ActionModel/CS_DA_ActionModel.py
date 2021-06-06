from Lib.Gaussian import Gaussian
class CS_DA_ActionModel:
    def __init__(self, S=None, A=None, mu_a=None, Sigma_a=None, AM=None):
        if AM is None:
            self.S = S
            self.A = A
            na = len(self.A.values)
            self.gA = []
            for i in range(na):
                self.gA.append(Gaussian(mu_a[i], Sigma_a[i]))
        else:
            self.S = AM.S
            self.A = AM.A
            self.gA = AM.gA

    def Prediction(self, s, b, a, Sp):
        """
        Belief evolution under the given action a.
        :param s: The hidden state. We also update it to use it in simulations.
        Note, however, that this state is never accessible by the planner
        (only the beliefs are).
        :param b: The belief to update.
        :param a: The action to apply
        :param Sp: Space where the belief is defined (used to bound the belief  and
        the hidden state if necessary).
        :return: The updated hidden state and the updated belief.
        """

        bOut, t = self.BeliefPrediction(b, a, Sp)
        sOut = Sp.Crop(s+t.rand())
        return sOut, bOut

    def BeliefPrediction(self, b, a, Sp):
        """
        Belief evolution under a given action.
        """

        t = self.GetActionModelFixedA(a)
        bOut = b.Prediction(t, Sp)
        return bOut, t

    def GetActionModelFixedA(self, a):
        """
        Returns the action model for a given action a.
        :param a:
        :return: a Gaussian with the mean and covariance.
        """
        return self.gA[int(a-1)]

