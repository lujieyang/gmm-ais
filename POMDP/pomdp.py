import numpy as np
import copy
from ActionModel.CS_DA_ActionModel import CS_DA_ActionModel
from ObsModel.CS_DO_ObsModel import CS_DO_ObsModel
from RewardModel.CS_DA_RewardModel import CS_DA_RewardModel
from Lib.utils import *

class POMDP:
    def __init__(self, gamma, name="Test"):
        self.name = name
        self.gamma = gamma



class CS_POMDP(POMDP):
    def __init__(self, gamma, ncAlpha, name):
        super().__init__(gamma, name)
        self.maxAlphaC = ncAlpha

class CS_DO_POMDP(CS_POMDP):
    def __init__(self, gamma, ncAlpha, name):
        super().__init__(gamma, ncAlpha, name)

class CS_DO_DA_POMDP(CS_DO_POMDP, CS_DA_ActionModel, CS_DO_ObsModel, CS_DA_RewardModel):
    def __init__(self, S, A, O, AM, OM, RM, gamma, ncAlpha, name="Test"):
        CS_DO_POMDP.__init__(self, gamma, ncAlpha, name)
        CS_DA_ActionModel.__init__(self, AM=AM)
        CS_DO_ObsModel.__init__(self, S, O, OM.p)
        CS_DA_RewardModel.__init__(self, S, A, RM.r)
        self.S = S
        self.A = A
        self.O = O

    def SampleBeliefs(self, start, nBeliefs, minBeliefDist, stepsXtrial, minR, maxR):
        np.random.seed(8888)
        A = self.A
        S = self.S

        md = minBeliefDist + 1
        B = []
        BO = []
        s_s = []
        a_s = []
        o_s = []
        r_s = []
        step_ind = []
        k = 0
        r = maxR - 1
        while k < nBeliefs:
            if (k % stepsXtrial == 0):  #or (r > maxR) or (r < minR):
                b = start
                s = S.Crop(b.rand())
                step_ind.append(k)

            a = A.rand()
            s, b, o, r, bn = self.SimulationStep(b, s, a)

            # if k > 4170:
            #     print("s")
            # b.plot()

            if (k > 1) and (minBeliefDist > 0):
                md = b.Distance(BO[k-1])

            if md > minBeliefDist:
                k += 1
                BO.append(b)
                B.append(bn)
                s_s.append(s)
                a_s.append(a)
                o_s.append(o)
                r_s.append(r)
                print(".", end=" ")
                if k % 80 == 0:
                    print("\n")
        print("\n")

        return BO, B, s_s, a_s, o_s, r_s, step_ind

    def SimulationStep(self, b, s, a):
        S = self.S

        s, b = self.Prediction(s, b, a, S)
        bn = copy.copy(b)
        po = self.GetObsModelFixedS(s)

        o = RandVector(po)
        b = self.Update(b, o, S)

        r = self.Reward(a, s)

        return s, b, o, r, bn










