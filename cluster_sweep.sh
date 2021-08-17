#!/bin/bash

#SBATCH --array=0-19
#SBATCH -o cluster_sweep.sh.log
#SBATCH --exclusive


a_values=( 50 75 100 200 250 300 450 550 600 650 700 750 800 850 900 950 1000 1100 1500)

trial=${SLURM_ARRAY_TASK_ID}
a=${a_values[$(( trial % ${#a_values[@]} ))]}
trial=$(( trial / ${#a_values[@]} ))

python src/clustering.py --nz $a --folder_name "data/100k/"