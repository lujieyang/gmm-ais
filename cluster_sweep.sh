#!/bin/bash

#SBATCH --array=0-190
#SBATCH -o cluster_sweep.sh.log
#SBATCH --exclusive
#SBATCH -c 8


a_values=( 50 75 100 200 250 300 450 400 500 550 600 650 700 750 800 850 900 950 1000 1250 1500 )
b_values=( 67 88 42 157 33 77 1024 2048 512 32 )

trial=${SLURM_ARRAY_TASK_ID}
a=${a_values[$(( trial % ${#a_values[@]} ))]}
trial=$(( trial / ${#a_values[@]} ))
b=${b_values[$(( trial % ${#b_values[@]} ))]}

python src/clustering.py --nz $a --seed $b
