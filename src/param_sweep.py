import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, default="model/")
    args = parser.parse_args()

    nz_list = [50, 100, 200, 250, 500, 750, 1000]
    tau_list = [1, 5, 10, 50, 100]
    lr_list = [1e-3, 3e-3, 1e-4, 3e-4]
    seed = 42
    nf = 96

    n_lr = len(lr_list)
    n_nz = len(nz_list)
    n_tau = len(tau_list)
    avg_mean = np.zeros((n_lr, n_tau, n_nz))
    avg_std = np.zeros((n_lr, n_tau, n_nz))
    for n1 in range(n_lr):
        lr = lr_list[n1]
        folder_name = args.folder_name + "seed" + str(seed) + "/lr" + str(lr) + "/"
        for n2 in range(n_tau):
            tau = tau_list[n2]
            for n3 in range(n_nz):
                nz = nz_list[n3]
                avg_mean[n1, n2, n3] = np.load(folder_name + "mean_{}_{}_{}.npy".format(nz, nf, tau))
                avg_std[n1, n2, n3] = np.load(folder_name + "std_{}_{}_{}.npy".format(nz, nf, tau))
            plt.errorbar(nz_list, avg_mean[n1, n2, :], avg_std[n1, n2, :], linestyle='None', fmt='-o', label=str(tau))
        plt.xticks(nz_list)
        plt.legend(loc="best")
        plt.title("Lr = " + str(lr))
        plt.show()

    file_name = args.folder_name + "performance.csv"

    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([args.folder_name])
        for i in range(n_lr):
            csvwriter.writerow(["lr = " + str(lr_list[i])])
            csvwriter.writerow(tau_list)
            for j in range(n_nz):
                csvwriter.writerow(avg_mean[i, :, j])

