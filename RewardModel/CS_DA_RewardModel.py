
class CS_DA_RewardModel:
    def __init__(self, S, A, r):
        self.S = S
        self.A = A
        self.r = r

    def Reward(self, a, s):
        """
        Evaluates the reward function.
        :param a:
        :param s:
        :return:
        """
        return self.r[int(a)-1].Value(s) #* 10
