import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from clustering import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_data", help="Load belief samples", action="store_true")
    parser.add_argument("--calculate_loss", action="store_true")
    parser.add_argument("--data_folder", help="Folder name for data", type=str, default="data/p0/")
    parser.add_argument("--folder_name", type=str, default="cluster/p0/")
    args = parser.parse_args()

    nz_list = [50, 75, 100, 200, 250, 300, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500]
    nb = 1000
    nu = 3
    seeds = [67, 88, 42, 157, 33, 77, 1024, 2048, 512, 32]

    if args.load_data:
        bt, b_next, bp, st, s_next, action_indices, reward, reward_b = load_data(folder_name=args.data_folder)

    avg_mean = []
    avg_std = []
    dt = []
    loss_s = []
    for nz in nz_list:
        # avg_mean.append(np.load(args.folder_name + "mean_{}_{}.npy".format(nz, nb))/10)
        # avg_std.append(np.load(args.folder_name + "std_{}_{}.npy".format(nz, nb))/np.sqrt(10))
        aR = []
        time = []
        loss = []
        for seed in seeds:
            aR.append(np.load(args.folder_name + "aR_{}_{}.npy".format(nz, seed))/10)
            if args.calculate_loss:
                B, r, kmeans = load_model(nz, seed, args.folder_name)
                l = calculate_loss(bt, bp, reward, action_indices, nz, nu, B, r, kmeans)
                loss.append(l)
            time.append(np.load(args.folder_name + "time_{}_{}.npy".format(nz, seed))/np.sqrt(10))
        aR = np.array(aR)
        avg_mean.append(np.mean(aR))
        avg_std.append(np.std(aR))
        dt.append(np.mean(time))
        loss_s.append(np.mean(loss))
    colors = cm.rainbow(np.linspace(0, 1, len(nz_list)))
    plt.errorbar(nz_list, avg_mean, avg_std, linestyle='None', fmt='-o', ecolor=colors)
    # plt.xticks(nz_list)
    # plt.legend(loc="best")
    plt.xlabel('Number of AIS')
    plt.ylabel('Average Return')
    plt.show()

    file_name = args.folder_name + "performance.csv"

    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(nz_list)
        csvwriter.writerow(avg_mean)
        csvwriter.writerow(avg_std)
        csvwriter.writerow(dt)
        csvwriter.writerow(loss_s)

