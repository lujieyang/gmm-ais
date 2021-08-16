import numpy as np
import matplotlib.pyplot as plt

from Space.CSpace import CSpace
from Space.DSpace import DSpace
from Belief.belief import GBelief
from ActionModel.CS_DA_ActionModel import CS_DA_ActionModel
from ObsModel.CS_DO_ObsModel import CS_DO_ObsModel
from RewardModel.CS_DA_RewardModel import CS_DA_RewardModel
from POMDP.pomdp import CS_DO_DA_POMDP
from Lib.GMixture import GMixture
from Lib.Gaussian import Gaussian

def GetTest1Parameters(ncBelief=4, ncAlpha=9, actionScale=2):
    """
    :param ncBelief: Number of components in the belief mixtures.
    :param ncAlpha: Number of components in the alpha mixtures.
    :param actionScale: Scale factor to apply to the right/left displacements.
    :return:
    """

    # Define the POMDP
    # State space is 1-D in the range [-20,20]
    S = CSpace(-20, 20)
    # 3 actions: left, right, enterDoor
    A = DSpace(3)
    # 4 Observations: left-end, right-end, door, corridor
    O= DSpace(4)
    # discount factor
    gamma = 0.95

    # Action model with continuous states and discrete actions
    mu_a = [-actionScale, actionScale, 0]
    Sigma_a = [0.05, 0.05, 0.05]
    AM = CS_DA_ActionModel(S, A, mu_a, Sigma_a)

    # Observation model with continuous states and discrete observations.
    # Note that we actually define p(o,s) and that we assume p(s) to be
    # uniform. Thus Gaussians should be evenly distributed in 's' and with
    # the adequate covariance to define a uniform coverage.
    so = 1.6
    om = []
    # Left-end
    om.append(GMixture(np.ones((1, 5)), [Gaussian(-21, so), Gaussian(-19, so),
                                         Gaussian(-17, so), Gaussian(-15, so),
                                         Gaussian(-13, so)]))
    # Right-end
    om.append(GMixture(np.ones((1, 5)), [Gaussian(21, so), Gaussian(19, so),
                                         Gaussian(17, so), Gaussian(15, so),
                                         Gaussian(13, so)]))
    sd = 1.6
    # Door
    om.append(GMixture(np.ones((1, 4)), [Gaussian(-11, sd), Gaussian(-5, sd),
                                         Gaussian(3, sd), Gaussian(9, sd)]))
    sc = 1.6
    # Corridor
    om.append(GMixture(np.ones((1, 8)), [Gaussian(-9, sc), Gaussian(-7, sc),
                                         Gaussian(-3, sc), Gaussian(-1, sc),
                                         Gaussian(1, sc), Gaussian(5, sc),
                                         Gaussian(7, sc), Gaussian(11, sc)]))
    OM = CS_DO_ObsModel(S, O, om)

    # Reward model with continuous states and discrete actions
    rm = []
    rm.append(GMixture(np.array([-2, -2, -2]), [Gaussian(-21, 1), Gaussian(-19, 1), Gaussian(-17, 1)]))
    rm.append(GMixture(np.array([-2, -2, -2]), [Gaussian(21, 1), Gaussian(19, 1), Gaussian(17, 1)]))
    rm.append(GMixture(np.array([-10, 2, -10]), [Gaussian(-25, 250), Gaussian(3, 3), Gaussian(25, 250)]))
    RM = CS_DA_RewardModel(S, A, rm)

    # Assemble the POMDP
    POMDP = CS_DO_DA_POMDP(S, A, O, AM, OM, RM, gamma, ncAlpha, 'Test1')

    # Define the parameters for sampling beliefs
    # Define the start belief
    g1 = Gaussian(-15, 30)
    P = {}
    P["start"] = GBelief(GMixture(np.array([1, 1, 1, 1]), [g1, g1 + 10, g1 + 20, g1 + 30]), ncBelief)
    # Belief sampling
    P["nBeliefs"] = 500
    P["dBelief"] = 0  # 0.1
    P["stepsXtrial"] = 30
    P["rMin"] = -0.5
    P["rMax"] = 0.5
    # Testing
    P["maxTime"] = 2500
    P["stTime"] = 100
    P["numTrials"] = 100
    # Parameters for solving
    P["stopCriteria"] = lambda n, t, vc: (t > P["maxTime"])

    return POMDP, P


def plot_model(gms, title):
    for gm in gms:
        gm.plot_()
    axes = plt.gca()
    axes.set_xlim([-21, 21])
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    test = "Test1"
    POMDP, P = GetTest1Parameters(ncBelief=10)
    plot_model(POMDP.p, "Observation Model")
    plot_model(POMDP.r, "Reward Model")
    BO, B, s, a, o, r, P_o_ba, step_ind = POMDP.SampleBeliefs(P["start"], P["nBeliefs"], P["dBelief"],
                                     P["stepsXtrial"], P["rMin"], P["rMax"])