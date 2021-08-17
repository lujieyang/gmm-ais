#!/bin/bash

#SBATCH --array=0-196
#SBATCH -o eval_sweep.sh.log
#SBATCH --gres=gpu:volta:2


a_values=( 50 100 200 250 500 750 1000 )
b_values=( 1 5 10 50 100 200 500)
c_values=( 1e-3 3e-3 1e-4 3e-4)

trial=${SLURM_ARRAY_TASK_ID}
a=${a_values[$(( trial % ${#a_values[@]} ))]}
trial=$(( trial / ${#a_values[@]} ))
b=${b_values[$(( trial % ${#b_values[@]} ))]}
trial=$(( trial / ${#b_values[@]} )
c=${c_values[$(( trial % ${#c_values[@]} ))]}

python src/eval.py --nz $a --tau $b --lr $c --folder_name "model/6_layer/"