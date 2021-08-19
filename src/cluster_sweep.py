import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, default="cluster/")
    args = parser.parse_args()

    nz_list = [50, 75, 100, 200, 250, 300, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1500]
    nb = 1000

    avg_mean = []
    avg_std = []
    for nz in nz_list:
        avg_mean.append(np.load(args.folder_name + "mean_{}_{}.npy".format(nz, nb))/10)
        avg_std.append(np.load(args.folder_name + "std_{}_{}.npy".format(nz, nb))/np.sqrt(10))
    colors = cm.rainbow(np.linspace(0, 1, len(nz_list)))
    plt.errorbar(nz_list, avg_mean, avg_std, linestyle='None', fmt='-o', ecolor=colors)
    # plt.xticks(nz_list)
    # plt.legend(loc="best")
    plt.show()

    file_name = args.folder_name + "performance.csv"

    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(nz_list)
        csvwriter.writerow(avg_mean)
        csvwriter.writerow(avg_std)

