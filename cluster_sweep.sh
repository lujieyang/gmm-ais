#!/bin/bash

#SBATCH --array=0-159
#SBATCH -o cluster_sweep.sh.log
#SBATCH --exclusive
#SBATCH -c 8


a_values=( 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 )
b_values=( 67 88 42 157 33 77 1024 2048 512 32 )

trial=${SLURM_ARRAY_TASK_ID}
a=${a_values[$(( trial % ${#a_values[@]} ))]}
trial=$(( trial / ${#a_values[@]} ))
b=${b_values[$(( trial % ${#b_values[@]} ))]}

python src/clustering.py --nz $a --seed $b
