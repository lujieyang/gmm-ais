#!/bin/bash

#SBATCH --array=0-74
#SBATCH -o LCB.log
#SBATCH --exclusive
#SBATCH -c 8


a_values=($( seq 30 5 100 ))
# b_values=( 67 88 42 157 33 77 1024 2048 512 32 )
b_values=(.92 .95 .96 .98 .99)

trial=${SLURM_ARRAY_TASK_ID}
a=${a_values[$(( trial % ${#a_values[@]} ))]}
trial=$(( trial / ${#a_values[@]} ))
b=${b_values[$(( trial % ${#b_values[@]} ))]}

python src/clustering.py --nz $a --lmbda $b #--seed $b --reward_expectation
