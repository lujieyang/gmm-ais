import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from clustering import *

def plot_data_median(x, data, color, label):
    m, lo, hi = np.median(data, 1), \
            np.quantile(data, 0.25, 1), \
            np.quantile(data, 0.75, 1)
    plt.plot(x, m, '-', color=color, alpha=0.8, label=label)
    plt.fill_between(x, lo, hi, color=color, alpha=0.2)

def get_data_median(x, data, color, label):
    m, lo, hi = np.median(data), \
                np.quantile(data, 0.25), \
                np.quantile(data, 0.75)
    nx = len(x)
    m = np.ones(nx)*m
    lo = np.ones(nx) * lo
    hi = np.ones(nx) * hi
    plt.plot(x, m, '--', color=color, alpha=0.8, label=label)
    plt.fill_between(x, lo, hi, color=color, alpha=0.2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", help="Load belief samples", action="store_true")
    parser.add_argument("--calculate_loss", action="store_true")
    parser.add_argument("--data_folder", help="Folder name for data", type=str, default="data/p0/")
    parser.add_argument("--folder_name", type=str, default="cluster/p0/")
    args = parser.parse_args()

    # nz_list = [50, 75, 100, 200, 250, 300, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500]
    nz_list = range(50, 101)
    nb = 1000
    nu = 3
    seeds = [67, 88, 42, 157, 33, 77, 1024, 2048, 512, 32]

    if args.load_data:
        bt, b_next, bp, st, s_next, action_indices, reward, reward_b = load_data(folder_name=args.data_folder)

    avg_mean = []
    avg_std = []
    dt = []
    loss_B = []
    loss_r = []
    for nz in nz_list:
        aR = []
        time = []
        lB = []
        lr = []
        for seed in seeds:
            aR.append(np.load(args.folder_name + "aR_{}_{}.npy".format(nz, seed)) / 10)
            if args.calculate_loss:
                B, r, kmeans = load_model(nz, seed, args.folder_name)
                l = calculate_loss(bt, bp, reward, action_indices, nz, nu, B, r, kmeans)
            else:
                lB.append(np.load(args.folder_name + "lB_{}_{}.npy".format(nz, seed)))
                lr.append(np.load(args.folder_name + "lr_{}_{}.npy".format(nz, seed)))
            time.append(np.load(args.folder_name + "time_{}_{}.npy".format(nz, seed)) / np.sqrt(10))
        aR = np.array(aR)
        avg_mean.append(np.mean(aR, 1))
        avg_std.append(np.std(aR))
        dt.append(np.mean(time))
        loss_B.append(np.mean(lB))
        loss_r.append(np.std(lr))
    avg_mean = np.array(avg_mean)
    PPO_return = np.mean(np.load("model/PPO_return.npy"), 1)
    A2C_return = np.mean(np.load("model/A2C_return.npy"), 1)
    plot_data_median(nz_list, avg_mean, None, "DAIS")
    get_data_median(nz_list, PPO_return, "orange", "PPO")
    get_data_median(nz_list, A2C_return, "green", "A2C")
    # plt.errorbar(nz_list, avg_mean, avg_std, linestyle='None', fmt='-o', ecolor=colors)
    # plt.xticks(nz_list)
    plt.legend(loc="best", fontsize=16)
    plt.xlabel('DAIS Dimension', fontsize=20)
    plt.ylabel('Average Return', fontsize=20)
    # plt.title("Performance vs DAIS Dimension")
    plt.savefig("cn_return")
    plt.show()

    loss_B = np.array(loss_B)
    loss_r = np.array(loss_r)
    plt.plot(nz_list, loss_r, label="r prediciton")
    plt.plot(nz_list, loss_B, label="B prediciton")
    # plt.fill_between(nz_list, loss_mean + loss_std, loss_mean - loss_std, alpha=0.5)
    # plt.errorbar(nz_list, avg_mean, avg_std, linestyle='None', fmt='-o', ecolor=colors)
    # plt.xticks(nz_list)
    # plt.legend(loc="best")
    plt.xlabel('DAIS Dimension', fontsize=20)
    plt.ylabel('DAIS fitting loss', fontsize=20)
    plt.legend(loc="best", fontsize=16)
    # plt.title("Performance vs DAIS Dimension")
    plt.savefig("cn_loss")
    plt.show()

    file_name = args.folder_name + "performance.csv"

    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(nz_list)
        csvwriter.writerow(avg_mean)
        csvwriter.writerow(avg_std)
        csvwriter.writerow(dt)
        csvwriter.writerow(loss_B)
        csvwriter.writerow(loss_r)

# Parameter 0
t = np.arange(25) * 100
pbvi_mean = np.array(
    [-4.1935, -3.8491, -5.0012, -6.0703, -6.3253, -3.0053, 0.0216, 1.2603, 1.2965, 1.3922, 1.3307, 1.4103, 1.4343,
     1.4116, 1.3827, 1.4163,
     1.4359, 1.4456, 1.4393, 1.4608, 1.4763, 1.5047, 1.5352, 1.5108, 1.4598])
pbvi_std = np.array(
    [1.6919, 0.8076, 0.8656, 0.5467, 0.3952, 3.4997, 2.4044, 0.1766, 0.2895, 0.1402, 0.2911, 0.3225, 0.1368, 0.1237,
     0.1587, 0.1504,
     0.0774, 0.1080, 0.1045, 0.1096, 0.1241, 0.1234, 0.1256, 0.1135, 0.1211])
# plt.errorbar(t, pbvi_mean, pbvi_std, linestyle='-', fmt='-o', ms=4)
plt.errorbar(353.408266564284, 1.54554443034196, 0.0823720658208118, fmt='-o', ms=4, label="DAIS")
plt.plot(t, pbvi_mean, label="CPBVI")
plt.fill_between(t, pbvi_mean + pbvi_std, pbvi_mean - pbvi_std, alpha=0.5, color="orange")
# plt.legend(["CPBVI", "DAIS"], loc="best", fontsize=12)
plt.legend(loc="best", fontsize=16)
plt.xlabel("time(s)", fontsize=20)
plt.ylabel("Average Return", fontsize=20)
# plt.title("DAIS vs CPBVI")
plt.savefig("cn_compare")
plt.show()

# Parameter 1
t = np.arange(50) * 100
pbvi_mean = [-6.5124, -7.3852, -9.1995, 3.0874, 8.7093, 8.0920, 6.7244, 7.9636, 7.5937, 7.8296, 5.5094, 6.0144, 7.9034,
             5.3638, 6.2347, 5.6153,
             5.6345, 5.1589, 5.6257, 5.3716, 4.4087, 4.0815, 3.8421, 4.0065, 3.7169, 4.5157, 3.0798, 3.8588, 3.4644,
             3.1697, 2.0389, 1.7265,
             2.4602, 2.2082, 1.6639, 1.4183, 1.3738, 1.9850, 0.7102, 0.7353, 1.3020, 0.8098, 1.1461, 1.2846, 1.0731,
             0.8385, 0.4301, 0.6798,
             1.1020, 1.0072]
pbvi_std = [4.5398, 2.1067, 3.6101, 7.1251, 1.8738, 1.2462, 1.6035, 1.1512, 1.6632, 1.7414, 1.8321, 2.0051, 1.8162,
            2.0398, 1.9404, 1.8895,
            1.6998, 1.8523, 1.9764, 1.9645, 1.6706, 1.6495, 2.0018, 2.2856, 2.0495, 2.3703, 2.6238, 2.1556, 2.5150,
            2.2780, 1.2113, 1.7328,
            2.5483, 1.7598, 1.5943, 1.0190, 1.6813, 0.8433, 0.8609, 0.6761, 0.9895, 0.7502, 0.7098, 1.2293, 1.1573,
            0.9617, 0.6674, 0.9300,
            1.7632, 1.1993]
plt.errorbar(t, pbvi_mean, pbvi_std, linestyle='-', fmt='-o', ms=4)
ind = np.argmax(avg_mean)
plt.errorbar(dt[ind], avg_mean[ind], avg_std[ind], fmt='-o', ms=4)
plt.legend(["CPBVI", "DAIS"], loc="best")
plt.xlabel("time(s)")
plt.ylabel("Average Return")
plt.title("DAIS vs CPBVI")
plt.show()