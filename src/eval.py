import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.nn import functional as F
import torch.nn as nn
from Experiments.GetTestParameters import GetTest1Parameters
from src.train_map_dynamics import process_belief, load_model

def value_iteration(B, r, nz, na, z_list, epsilon=0.0001, discount_factor=0.95):
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
        reward = r[a](torch.from_numpy(z_one_hot).to(torch.float32)).detach().numpy()
        v = reward + discount_factor * z_next@V

        return v

    # start with initial value function and initial policy
    V = np.zeros(nz)
    policy = np.zeros([nz, na])

    n = 0
    # while not the optimal policy
    while True:
        # for stopping condition
        delta = 0

        if n % 100 == 0:
            print("Value Iteration: " + str(n))
        # loop over state space
        for z in z_list:

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

    print("Value Iteration: Done")
    return policy, V


def eval_performance(policy, V, POMDP, start, D, na, tau=1, B_det=None, n_episodes=100, beta=0.95):
    returns = []
    Vs = []
    S = POMDP.S
    for n_eps in range(n_episodes):
        # if n_eps % 10 == 0:
        #     print('Epoch :', n_eps)
        reward_episode = []
        b = copy.deepcopy(start)
        s = S.Crop(b.rand())
        z_one_hot = F.gumbel_softmax(D(torch.from_numpy(b.to_array()).to(torch.float32)), tau=tau, hard=True).data.numpy()
        # z_one_hot = F.gumbel_softmax(D(torch.from_numpy(np.array(s).reshape(-1)).to(torch.float32)), hard=True).data.numpy()

        for j in range(30):
            ind_z = np.where(z_one_hot == 1)[0][0]

            Vs.append(V[ind_z])

            try:
                action = np.arange(na)[policy[ind_z].astype(bool)][0]
            except:
                action = np.random.randint(na)

            s, b, o, r, rb , _, _ = POMDP.SimulationStep(b, s, action+1)
            reward_episode.append(r)

            if B_det is not None:
                z_one_hot = B_det@z_one_hot
            else:
                z_one_hot = F.gumbel_softmax(D(torch.from_numpy(b.to_array()).to(torch.float32)), tau=tau, hard=True).data.numpy()
                # z_one_hot = F.gumbel_softmax(D(torch.from_numpy(np.array(s).reshape(-1)).to(torch.float32)),
                #                              hard=True).data.numpy()

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


def interpret(BO, s, a, o, reward, rb, D, r, nu, tau=1):
    num_samples = len(BO)
    dict = {}
    d = {}
    for j in range(nu):
        d[j] = []
    for i in range(num_samples):
        b = BO[i]
        z = F.gumbel_softmax(D(torch.from_numpy(b.to_array()).to(torch.float32)), tau=tau, hard=True)
        # z = F.gumbel_softmax(D(torch.from_numpy(np.array(s[i]).reshape(-1)).to(torch.float32)), hard=True)
        z_one_hot = z.data.numpy()
        z_cluster = np.where(z_one_hot == 1)[0][0]
        a_ind = int(a[i]-1)
        if z_cluster not in dict.keys():
            dict[z_cluster] = {}
            dict[z_cluster]["s"] = copy.deepcopy(d)
            dict[z_cluster]["o"] = copy.deepcopy(d)
            dict[z_cluster]["r"] = copy.deepcopy(d)
            dict[z_cluster]["rb"] = copy.deepcopy(d)
            dict[z_cluster]["b"] = copy.deepcopy(d)
            dict[z_cluster]["r_pred"] = copy.deepcopy(d)
        dict[z_cluster]["s"][a_ind].append(s[i])
        dict[z_cluster]["o"][a_ind].append(o[i])
        dict[z_cluster]["r"][a_ind].append(reward[i])
        dict[z_cluster]["rb"][a_ind].append(rb[i])
        dict[z_cluster]["b"][a_ind].append(b)
        dict[z_cluster]["r_pred"][a_ind].append(r[int(a[i] - 1)](z).detach().numpy()[0])

    for c in dict.keys():
        for j in range(nu):
            r_pred = np.array(dict[c]["r_pred"][j])
            if (r_pred > 1).any():
                plt.plot(dict[c]["s"][j], dict[c]["r"][j], 'rx')
                plt.plot(dict[c]["s"][j], dict[c]["rb"][j], 'bo')
                plt.plot(dict[c]["s"][j], dict[c]["r_pred"][j], 'k.')
                plt.xlabel("s")
                plt.ylabel("r")
                plt.title("Reward Prediction with Action {} for AIS {}".format(j, c))
                plt.show()
    return dict


def minimize_B(z_list, B, nz):
    remove_list = np.array(list(set(range(nz)) - set(z_list)))
    B[:, remove_list, :] = 0
    B[:, :, remove_list] = 0
    # B_sum = np.sum(B, axis=1).reshape((B.shape[0], 1, B.shape[1]))
    # B_sum[:, 0, remove_list] = 1
    return B #/B_sum


def B_det_to_prob(B_det, nu, no):
    B = []
    for i in range(nu):
        B.append(np.sum(B_det[i*no:(i+1)*no, :, :], axis=0))
    return np.array(B)


def validation_loss(B, r, D, loss_fn_z, loss_fn_r, nu, bt, bp, b_next, reward, action_indices, tau=1):
    # Validation Loss
    pred_loss = 0
    r_loss = 0
    for i in range(nu):
        # Calculate loss for each (discrete) action
        ind = (np.array(action_indices) == i)
        Db = F.gumbel_softmax(D(bt[ind]), tau=tau, hard=True)
        z = B[i]@Db.detach().numpy().T
        z_next = F.gumbel_softmax(D(bp[ind]), tau=tau, hard=True)
        r_pred = r[i](Db)
        pred_loss += loss_fn_z(torch.from_numpy(z.T).to(torch.float32), z_next)
        r_loss += loss_fn_r(r_pred, reward[ind])
    print("Prediction loss: {}, reward loss: {}".format(pred_loss, r_loss))


def plot_reward(dict, z_list, nu):
    for i in range(nu):
        for z in dict.keys():
            l = len(dict[z]["rb"][i])
            plt.plot(z * np.ones(l), dict[z]["rb"][i], "kx")
            plt.plot(z * np.ones(l), dict[z]["r_pred"][i], "r.")
        plt.title("Expected Reward Prediction for Action {}".format(i))
        plt.show()


if __name__ == '__main__':
    nz = 1000
    nu = 3
    no = 4
    nf = 96
    ncBelief = 10
    tau = 1
    AP2ab = False
    POMDP, P = GetTest1Parameters(ncBelief=ncBelief)
    B, r, D, z_list = load_model(nz, nf, nu, tau, AP2ab=AP2ab)

    if AP2ab:
        B = B_det_to_prob(B, nu, no)
    num_samples = 5000
    BO, BS, s, a, o, reward, rb, P_o_ba, step_ind = POMDP.SampleBeliefs(P["start"], num_samples, P["dBelief"],
                                                      P["stepsXtrial"], P["rMin"], P["rMax"])
    dict = interpret(BO, s, a, o, reward, rb, D, r, nu, tau=tau)
    plot_reward(dict, z_list, nu)
    B = minimize_B(z_list, B, nz)
    print("Minimized number of AIS: ", len(z_list))

    bt, b_next, bp, st, s_next, input_dim, g_dim, action_indices, observation_indices, reward_values, P_o_ba_t = \
        process_belief(BO, BS, num_samples, step_ind, ncBelief, s, a, o, rb, P_o_ba)
    st_ = torch.from_numpy(st).view(st.shape[0], 1).to(torch.float32)
    s_next_ = torch.from_numpy(s_next).view(bt.shape[0], 1).to(torch.float32)
    bt_ = torch.from_numpy(bt).to(torch.float32)
    bp_ = torch.from_numpy(bp).to(torch.float32)
    b_next_ = torch.from_numpy(b_next).to(torch.float32)
    r_ = torch.from_numpy(reward_values).view(st.shape[0], 1).to(torch.float32)
    loss_fn_z = nn.L1Loss()
    loss_fn_r = nn.MSELoss()
    validation_loss(B, r, D, loss_fn_z, loss_fn_r, nu, bt_, bp_, b_next_, r_, action_indices, tau=tau)

    policy, V = value_iteration(B, r, nz, nu, z_list)
    aR = []
    for i in range(100):
        print("Trial ", i)
        aR.append(eval_performance(policy, V, POMDP, P["start"], D, nu, tau=tau))
    print("Average reward: ", np.mean(np.array(aR)))


