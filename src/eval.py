import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from Experiments.GetTestParameters import GetTest1Parameters
from src.train_map_dynamics import *

def value_iteration(B, r, nz, na, epsilon=0.0001, discount_factor=0.95):
    """
    Value Iteration Algorithm.

    Args:
        B: numpy array of size(na, nz, nz). transition probabilities of the environment P(z(t+1)|z(t), a(t)).
        r: numpy array of size (na, nz). reward function r(z(t),a(t))
        nz: number of AIS in the environment.
        na: number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(V, a, z):
        z_one_hot = np.zeros(nz)
        z_one_hot[z] = 1
        z_next = B[a, :, :]@z_one_hot
        # ind = (z_next > epsilon)
        v = r[a, :]@z_one_hot + discount_factor * z_next@V

        return v

    # start with initial value function and initial policy
    V = np.zeros(nz)
    policy = np.zeros([nz, na])

    n = 0
    # while not the optimal policy
    while True:
        # for stopping condition
        delta = 0

        # loop over state space
        for z in range(nz):

            actions_values = np.zeros(na)

            # loop over possible actions
            for a in range(na):
                # apply bellman eqn to get actions values
                actions_values[a] = one_step_lookahead(V, a, z)

            # pick the best action
            best_action_value = max(actions_values)

            # get the biggest difference between best action value and our old value function
            delta = max(delta, abs(best_action_value - V[z]))

            # apply bellman optimality eqn
            V[z] = best_action_value

            # to update the policy
            best_action = np.argmax(actions_values)

            # update the policy
            policy[z] = np.eye(na)[best_action]


        # if optimal value function
        if (delta < epsilon):
            break
        n += 1

    return policy, V


def eval_performance(policy, V, POMDP, start, D, na, B_det=None, n_episodes=100, beta=0.95):
    returns = []
    Vs = []
    # V_bs = []
    S = POMDP.S
    for n_eps in range(n_episodes):
        reward_episode = []
        b = start
        s = S.Crop(b.rand())
        z_one_hot = F.gumbel_softmax(D(torch.from_numpy(b.to_array()).to(torch.float32)), hard=True).data.numpy()

        for j in range(50):
            ind_z = np.where(z_one_hot == 1)[0][0]

            Vs.append(V[ind_z])

            action = np.arange(na)[policy[ind_z].astype(bool)][0]

            s, b, o, r, bn = POMDP.SimulationStep(b, s, action)
            reward_episode.append(r)

            if B_det is not None:
                z_one_hot = B_det@z_one_hot
            else:
                z_one_hot = F.gumbel_softmax(D(torch.from_numpy(b.to_array()).to(torch.float32)), hard=True).data.numpy()

        rets = []
        R = 0
        for i, r in enumerate(reward_episode[::-1]):
            R = r + beta * R
            rets.insert(0, R)
        returns.append(rets[0])

    average_return = np.mean(returns)
    # V_mse = np.linalg.norm(np.array(Vs)-np.array(V_bs))
    print("Average reward: ", average_return)
    # print("V mse: ", V_mse)
    # print("Average V mse", V_mse/len(Vs))
    return average_return  # , V_mse/len(Vs)


def interpret(POMDP, P, D, r):
    num_samples = 100
    BO, BS, s, a, o, reward, step_ind = POMDP.SampleBeliefs(P["start"], num_samples, P["dBelief"],
                                                      P["stepsXtrial"], P["rMin"], P["rMax"])
    bt, b_next, bp, st, s_next, input_dim, g_dim, action_indices, observation_indices, reward = \
        process_belief(BO, BS, num_samples, step_ind, ncBelief, s, a, o, r)
    for i in range(num_samples):
        print("s: {}, a: {}, o: {}, r: {}".format(s[i], a[i], o[i], reward[i]))
        BO[i].plot()
        z_one_hot = F.gumbel_softmax(D(torch.from_numpy(BO[i].to_array()).to(torch.float32)), hard=True).data.numpy()
        print("AIS cluster: ", np.where(z_one_hot==1)[0])
        print("Reward prediction: ", r[int(a[i])-1]@z_one_hot)


def load_model(nz, nf, nu):
    folder_name = "model/"
    B = np.load(folder_name + "B_{}_{}.npy".format(nz, nf))
    # r = np.load(folder_name + "r_{}_{}.npy".format(nz, nf))
    r_dict = torch.load(folder_name + "r_{}_{}.pth".format(nz, nf))
    r = []
    for i in range(nu):
        r.append(r_dict["model_" + str(i)])
        r[i].load_state_dict(r_dict[str(i)])
        r[i].eval()
    D = torch.load(folder_name + "D_pre_{}_{}_model.pth".format(nz, nf))
    D.load_state_dict(torch.load(folder_name + "D_pre_{}_{}.pth".format(nz, nf)))
    D.eval()
    return B, r, D


if __name__ == '__main__':
    POMDP, P = GetTest1Parameters()

    nz = 30
    nu = 3
    nf = 96
    ncBelief = 4
    B, r, D = load_model(nz, nf, nu)

    interpret(POMDP, P, D, r)
    policy, V = value_iteration(B, r, nz, nu)
    eval_performance(policy, V, POMDP, P["start"], D, nu)


