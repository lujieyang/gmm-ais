import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from sklearn.cluster import KMeans
import copy
import cvxpy as cp
from Experiments.GetTestParameters import *


def process_belief(BO, B, num_samples, step_ind, ncBelief, s, a, o, r, P_o_ba):
    step_ind_copy = np.array(step_ind)
    # Remove the first belief before observation (useless)
    B.pop(0)
    g_dim = BO[0].g[0].dim
    bt = []
    b_next = []
    b_next_p = []
    st = []
    s_next = []
    action_indices = []
    reward = []
    if g_dim == 1:
        for i in range(num_samples):
            b = 0 #BO[i].sample_array()
            # Belief before observation
            if i < num_samples-1:
                bp = 0#B[i].sample_array()
            if i in step_ind_copy-1:
                if i not in step_ind_copy:
                    b_next.append(b)
            else:
                bt.append(b)
                st.append(s[i])
                action_indices.append(int(a[i] - 1))
                reward.append(r[i])
                if i < num_samples - 1:
                    b_next_p.append(bp)
                    s_next.append(s[i+1])
                if i > 0 and i not in step_ind_copy:
                    b_next.append(bt[-1])
    else:
        pass

    return np.array(bt[:-1]), np.array(b_next), np.array(b_next_p), np.array(st[:-1]), np.array(s_next), \
           np.array(action_indices[:-1]), np.array(reward[:-1])


def cluster_state(st, s_next, reward, action_indices, nz, nu):
    X = st.reshape((len(st), 1))
    kmeans = KMeans(n_clusters=nz, random_state=0).fit(X)
    # l = st.astype(int)+20
    X_next = s_next.reshape((len(s_next), 1))
    pl = kmeans.predict(X_next) #s_next.astype(int)+20
    B = []
    r = []
    for i in range(nu):
        ind = (action_indices == i)
        l1 = kmeans.labels_[ind] #l[ind]
        z1 = np.zeros((nz, l1.size))
        z1[l1, np.arange(l1.size)] = 1
        l2 = pl[ind]
        z2 = np.zeros((nz, l2.size))
        z2[l2, np.arange(l2.size)] = 1
        B0 = LS_B(z1, z2) #solve_B(z1, z2, nz)
        B.append(B0)
        rT = np.linalg.lstsq(z1.T, reward[ind].T)
        r.append(rT[0].T)
    return np.array(B), np.array(r), kmeans


def LS_B(z1, z2):
    BT = np.linalg.lstsq(z1.T, z2.T)
    B0 = BT[0].T
    B0 = np.clip(B0, 0, 1)
    # s = np.sum(B0, axis=0)
    # s[s==0] = 1
    # B0 = B0/s
    return B0


def solve_B(z1, z2, nz):
    B = cp.Variable((nz, nz), nonneg=True)

    loss = cp.norm(cp.matmul(B, z1)-z2)
    constraints = [cp.matmul(np.ones((1, nz)), B) == 1]

    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, constraints)

    # solve problem
    problem.solve(solver=cp.SCS, verbose=True)

    if not (problem.status == cp.OPTIMAL):
        print("unsuccessful...")
    else:
        print("loss ", loss.value)
        pass

    return B.value


def grid_state(POMDP, nz, nu):
    B = []
    r = []
    for i in range(nu):
        x = np.linspace(-20, 20, nz)
        r.append(POMDP.r[i].Value(x))
        B0 = np.zeros((nz, nz))
        for j in range(nz):
            B0[:, j] = POMDP.gA[i].Value(x-x[j])
        B0 = B0/np.sum(B0, axis=0)
        B.append(B0)
    return np.array(B), np.array(r)


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
        reward = r[a][z]
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
        for z in np.arange(nz):

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


def eval_performance(policy, V, POMDP, start, na, B_det=None, n_episodes=100, beta=0.95):
    returns = []
    Vs = []
    S = POMDP.S
    for n_eps in range(n_episodes):
        reward_episode = []
        b = copy.deepcopy(start)
        s = S.Crop(b.rand())

        for j in range(30):
            ind_z = int(min([np.round((s+20)*nz/40), nz-1])) #

            Vs.append(V[ind_z])

            try:
                action = np.arange(na)[policy[ind_z].astype(bool)][0]
            except:
                action = np.random.randint(na)

            s, b, o, r, _, _ = POMDP.SimulationStep(b, s, action+1)
            reward_episode.append(r)

            if B_det is not None:
                z_one_hot = B_det@z_one_hot
            else:
                # z_one_hot = F.gumbel_softmax(D(torch.from_numpy(b.to_array()).to(torch.float32)), tau=tau, hard=True).data.numpy()
                pass

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


def save_data(bt, b_next, bp, st, s_next, action_indices, reward):
    folder_name = "data/"
    np.save(folder_name + "bt", bt)
    np.save(folder_name + "b_next", b_next)
    np.save(folder_name + "bp", bp)
    np.save(folder_name + "st", st)
    np.save(folder_name + "s_next", s_next)
    np.save(folder_name + "action_indices", action_indices)
    np.save(folder_name + "reward", reward)


def load_data(folder_name="data/"):
    bt = np.load(folder_name + "bt.npy")
    b_next = np.load(folder_name + "b_next.npy")
    bp = np.load(folder_name + "bp.npy")
    st = np.load(folder_name + "st.npy")
    s_next = np.load(folder_name + "s_next.npy")
    action_indices = np.load(folder_name + "action_indices.npy")
    reward = np.load(folder_name + "reward.npy")
    return bt, b_next, bp, st, s_next, action_indices, reward


def save_model(B_model, r_model, D_pre_model, z_list, nz, nf, tau, B_det_model=None, P_o_za_model=None):
    folder_name = "model/" + "100k/"
    r_dict = {}
    if B_det_model is not None:
        folder_name += "AP2ab/" + "obs_l_weight_2/"
        B_det = []
        for i in range(len(B_det_model)):
            B_det_model[i].cpu()
            B_det.append(B_det_model[i].weight.data.numpy())
        np.save(folder_name + "B_det_{}_{}_{}".format(nz, nf, tau), B_det)
        for j in range(len(r_model)):
            r_model[j].cpu()
            r_dict[str(j)] = r_model[j].state_dict()
            r_dict["model_" + str(j)] = r_model[j]
    else:
        B = []
        for i in range(len(B_model)):
            B_model[i].cpu()
            B.append(B_model[i].weight.data.numpy())
            r_model[i].cpu()
            r_dict[str(i)] = r_model[i].state_dict()
            r_dict["model_" + str(i)] = r_model[i]
        np.save(folder_name + "B_{}_{}_{}".format(nz, nf, tau), B)
    if P_o_za_model is not None:
        P_o_za = []
        for i in range(len(P_o_za_model)):
            P_o_za_model[i].cpu()
            P_o_za.append(P_o_za_model[i].weight.data.numpy())
        np.save(folder_name + "P_o_za_{}_{}_{}".format(nz, nf, tau), P_o_za)

    np.save(folder_name + "zList_{}_{}_{}".format(nz, nf, tau), z_list)
    torch.save(r_dict, folder_name + "r_{}_{}_{}.pth".format(nz, nf, tau))
    torch.save(D_pre_model.state_dict(), folder_name + "D_pre_{}_{}_{}.pth".format(nz, nf, tau))
    torch.save(D_pre_model, folder_name + "D_pre_{}_{}_{}_model.pth".format(nz, nf, tau))


def load_model(nz, nf, nu, tau, AP2ab=False):
    folder_name = "model/" + "100k/" + "scheduler/"
    if AP2ab:
        B = np.load(folder_name + "B_det_{}_{}_{}.npy".format(nz, nf, tau))
    else:
        B = np.load(folder_name + "B_{}_{}_{}.npy".format(nz, nf, tau))
    z_list = np.load(folder_name + "zList_{}_{}_{}.npy".format(nz, nf, tau))
    r_dict = torch.load(folder_name + "r_{}_{}_{}.pth".format(nz, nf, tau))
    r = []
    for i in range(nu):
        r.append(r_dict["model_" + str(i)])
        r[i].load_state_dict(r_dict[str(i)])
        r[i].eval()
    D = torch.load(folder_name + "D_pre_{}_{}_{}_model.pth".format(nz, nf, tau))
    D.load_state_dict(torch.load(folder_name + "D_pre_{}_{}_{}.pth".format(nz, nf, tau)))
    D.eval()
    return B, r, D, z_list


def plot_reward_value(kmeans, r, V, nu):
    x = np.linspace(-20, 20, 1000)
    l = kmeans.predict(x.reshape((1000, 1)))
    plt.plot(x, V[l], '.')
    plt.title("Value Function")
    plt.show()
    for i in range(nu):
        plt.plot(x, r[i, l], '.')
        plt.title("Reward Prediction")
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_trans", help="Fit the deterministic transition of AIS (AP2a)", action="store_true")
    parser.add_argument("--pred_obs", help="Predict the observation (AP2b)", action="store_true")
    parser.add_argument("--nz", help="Number of discrete AIS", type=int,
                        default=1000)
    parser.add_argument("--generate_data", help="Generate belief samples", action="store_true")
    args = parser.parse_args()

    # Sample belief states data
    ncBelief = 10
    POMDP, P = GetTest1Parameters(ncBelief=ncBelief)
    num_samples = 1000#0

    nz = args.nz
    nu = 3
    no = 4

    if args.generate_data:
        BO, BS, s, a, o, r, P_o_ba, step_ind = POMDP.SampleBeliefs(P["start"], num_samples, P["dBelief"],
                                                                   P["stepsXtrial"], P["rMin"], P["rMax"],
                                                                   obs_prob=args.pred_obs)
        bt, b_next, bp, st, s_next, action_indices, reward = \
            process_belief(BO, BS, num_samples, step_ind, ncBelief, s, a, o, r, P_o_ba)
        save_data(bt, b_next, bp, st, s_next, action_indices, reward)
    else:
        bt, b_next, bp, st, s_next, action_indices, reward = load_data()

    # B, r, kmeans = cluster_state(st, s_next, reward, action_indices, nz, nu)
    B, r = grid_state(POMDP, nz, nu)
    policy, V = value_iteration(B, r, nz, nu)
    # plot_reward_value(kmeans, r, V, nu)
    for i in range(50):
        eval_performance(policy, V, POMDP, P["start"], nu)
