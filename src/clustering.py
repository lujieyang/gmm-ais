import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import copy
import cvxpy as cp
import time
import pickle
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from Experiments.GetTestParameters import *


def process_belief(BO, B, num_samples, step_ind, nb, s, a, r, rb):
    step_ind_copy = np.array(step_ind)
    # Remove the first belief before observation (useless)
    B.pop(0)
    # State & belief collection is shifted by one time index from a, r, o, P_o_ba
    a.pop(0)
    r.pop(0)
    g_dim = BO[0].g[0].dim
    bt = []
    b_next = []
    b_next_p = []
    st = []
    s_next = []
    action_indices = []
    reward = []
    reward_b = []
    if g_dim == 1:
        for i in range(num_samples):
            b = BO[i].sample_array(nb=nb)
            # Belief before observation
            if i < num_samples-1:
                bp = B[i].sample_array(nb=nb)
            if i in step_ind_copy-1:
                if i not in step_ind_copy:
                    b_next.append(b)
            else:
                bt.append(b)
                st.append(s[i])
                if i < num_samples - 1:
                    b_next_p.append(bp)
                    s_next.append(s[i+1])
                    action_indices.append(int(a[i] - 1))
                    reward.append(r[i])
                    reward_b.append(rb[i])
                if i > 0 and i not in step_ind_copy:
                    b_next.append(bt[-1])
    else:
        pass

    return np.array(bt[:-1]), np.array(b_next), np.array(b_next_p), np.array(st[:-1]), np.array(s_next), \
           np.array(action_indices), np.array(reward), np.array(reward_b)


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
        B0 = solve_B(z1, z2, nz)
        B.append(B0)
        rT = np.linalg.lstsq(z1.T, reward[ind].T)
        r.append(rT[0].T)
    return np.array(B), np.array(r), kmeans


def cluster_belief(bt, bp, reward, action_indices, nz, nu):
    X = bt
    kmeans = KMeans(n_clusters=nz, random_state=0).fit(X)
    X_next = bp
    pl = kmeans.predict(X_next)
    B = []
    r = []
    for i in range(nu):
        ind = (action_indices == i)
        l1 = kmeans.labels_[ind]
        z1 = np.zeros((nz, l1.size))
        z1[l1, np.arange(l1.size)] = 1
        l2 = pl[ind]
        z2 = np.zeros((nz, l2.size))
        z2[l2, np.arange(l2.size)] = 1
        B0 = solve_B(z1, z2, nz)
        B.append(B0)
        rT = np.linalg.lstsq(z1.T, reward[ind].T)
        r.append(rT[0].T)
    return np.array(B), np.array(r), kmeans


def calculate_loss(bt, bp, reward, action_indices, nz, nu, B, r, kmeans):
    tl = kmeans.predict(bt)
    X_next = bp
    pl = kmeans.predict(X_next)
    loss_B = 0
    loss_r = 0
    for i in range(nu):
        ind = (action_indices == i)
        l1 = tl[ind]
        z1 = np.zeros((nz, l1.size))
        z1[l1, np.arange(l1.size)] = 1
        l2 = pl[ind]
        z2 = np.zeros((nz, l2.size))
        z2[l2, np.arange(l2.size)] = 1
        loss_B += mean_squared_error(B[i]@z1, z2)
        loss_r += mean_squared_error(r[i] @ z1, reward[ind])
    return loss_B, loss_r


def solve_B(z1, z2, nz):
    B = cp.Variable((nz, nz), nonneg=True)

    loss = cp.norm(cp.matmul(B, z1)-z2, "fro")
    constraints = [cp.matmul(np.ones((1, nz)), B) == 1]

    objective = cp.Minimize(loss)
    problem = cp.Problem(objective, constraints)

    # solve problem
    problem.solve(solver=cp.SCS, verbose=False)

    if not (problem.status == cp.OPTIMAL):
        print("unsuccessful...")
    else:
        print("loss ", loss.value)
        pass

    return B.value


def check_B(st, s_next, action_indices, nz, nu):
    l1 = np.clip(np.floor((st + 20) * nz / 40), 0, nz - 1).astype(int)
    l2 = np.clip(np.floor((s_next + 20) * nz / 40), 0, nz - 1).astype(int)
    for i in range(nu):
        ind = (action_indices == i)
        k1 = l1[ind]
        z1 = np.zeros((nz, k1.size))
        z1[k1, np.arange(k1.size)] = 1
        k2 = l2[ind]
        z2 = np.zeros((nz, k2.size))
        z2[k2, np.arange(k2.size)] = 1


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
            b_sample = b.sample_array(args.nb)
            ind_z = kmeans.predict(b_sample.reshape((1, args.nb)))[0] #int(min([np.floor((s+20)*nz/40), nz-1])) #

            Vs.append(V[ind_z])

            try:
                action = np.arange(na)[policy[ind_z].astype(bool)][0]
            except:
                action = np.random.randint(na)

            s, b, o, r, rb, _, _ = POMDP.SimulationStep(b, s, action+1)
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
    return average_return


def save_data(bt, b_next, bp, st, s_next, action_indices, reward, reward_b, folder_name="data/sample_belief/"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    np.save(folder_name + "bt", bt)
    np.save(folder_name + "b_next", b_next)
    np.save(folder_name + "bp", bp)
    np.save(folder_name + "st", st)
    np.save(folder_name + "s_next", s_next)
    np.save(folder_name + "action_indices", action_indices)
    np.save(folder_name + "reward", reward)
    np.save(folder_name + "reward_b", reward_b)


def load_data(folder_name="data/sample_belief/"):
    bt = np.load(folder_name + "bt.npy")
    b_next = np.load(folder_name + "b_next.npy")
    bp = np.load(folder_name + "bp.npy")
    st = np.load(folder_name + "st.npy")
    s_next = np.load(folder_name + "s_next.npy")
    action_indices = np.load(folder_name + "action_indices.npy")
    reward = np.load(folder_name + "reward.npy")
    reward_b = np.load(folder_name + "reward_b.npy")
    return bt, b_next, bp, st, s_next, action_indices, reward, reward_b


def save_model(B, r, kmeans, aR, dt, lB, lr, nz, seed, folder_name="cluster/"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    np.save(folder_name + "B_{}_{}".format(nz, seed), B)
    np.save(folder_name + "r_{}_{}".format(nz, seed), r)
    np.save(folder_name + "aR_{}_{}".format(nz, seed), aR)
    np.save(folder_name + "lB_{}_{}".format(nz, seed), lB)
    np.save(folder_name + "lB_{}_{}".format(nz, seed), lr)
    # np.save(folder_name + "std_{}_{}".format(nz, seed), np.std(np.array(aR)))
    np.save(folder_name + "time_{}_{}".format(nz, seed), dt)
    with open(folder_name + "kmeans_{}_{}.pkl".format(nz, seed), "wb") as f:
        pickle.dump(kmeans, f)


def load_model(nz, seed, folder_name):
    B = np.load(folder_name + "B_{}_{}.npy".format(nz, seed))
    r = np.load(folder_name + "r_{}_{}.npy".format(nz, seed))
    with open(folder_name + "kmeans_{}_{}.pkl".format(nz, seed), "rb") as f:
        kmeans = pickle.load(f)
    return B, r, kmeans


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
    parser.add_argument("--reward_expectation", help="Regression on ", action="store_true")
    parser.add_argument("--nz", help="Number of discrete AIS", type=int,
                        default=1000)
    parser.add_argument("--nb", help="Number of sample points to approximate belief distribution", type=int,
                        default=1000)
    parser.add_argument("--seed", help="Random seed", type=int, default=67)
    parser.add_argument("--num_samples", help="Number of Training Samples", type=int, default=100000)
    parser.add_argument("--generate_data", help="Generate belief samples", action="store_true")
    parser.add_argument("--data_folder", help="Folder name for data", type=str, default="data/p0/")
    parser.add_argument("--result_folder", help="Folder name for data", type=str, default="cluster/p0/")
    args = parser.parse_args()

    np.random.seed(args.seed)

    result_folder = args.result_folder

    # Sample belief states data
    ncBelief = 10
    POMDP, P = GetTestParameters0(ncBelief=ncBelief)
    num_samples = args.num_samples

    nz = args.nz
    nu = 3
    no = 4

    if args.generate_data:
        BO, BS, s, a, o, r, rb, P_o_ba, step_ind = POMDP.SampleBeliefs(P["start"], num_samples, P["dBelief"],
                                                                   P["stepsXtrial"], P["rMin"], P["rMax"],
                                                                   obs_prob=False)
        bt, b_next, bp, st, s_next, action_indices, reward, reward_b = \
            process_belief(BO, BS, num_samples, step_ind, args.nb, s, a, r, rb)
        save_data(bt, b_next, bp, st, s_next, action_indices, reward, reward_b, folder_name=args.data_folder)
    else:
        bt, b_next, bp, st, s_next, action_indices, reward, reward_b = load_data(folder_name=args.data_folder)

    start_time = time.time()
    if args.reward_expectation:
        B, r, kmeans = cluster_belief(bt, bp, reward_b, action_indices, nz, nu)
        result_folder = args.result_folder + "reward_expectation/"
    else:
        B, r, kmeans = cluster_belief(bt, bp, reward, action_indices, nz, nu)
    # B, r, kmeans = cluster_state(st, s_next, reward, action_indices, nz, nu)
    policy, V = value_iteration(B, r, nz, nu)
    # plot_reward_value(kmeans, r, V, nu)
    end_time = time.time()
    aR = []
    for i in range(30):
        aR.append(eval_performance(policy, V, POMDP, P["start"], nu))
    dt = end_time - start_time
    lB, lr = calculate_loss(bt, bp, reward, action_indices, nz, nu, B, r, kmeans)
    save_model(B, r, kmeans, aR, dt, lB, lr, nz, args.seed, folder_name=result_folder)
