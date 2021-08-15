#!/bin/bash

#SBATCH --array=0-14
#SBATCH -o param_sweep.sh.log-%j
#SBATCH --gres=gpu:volta:2


a_values=( 50 100 200 250 500 750 1000 )
b_values=( 50 100)
c_values=( 3e-3)

trial=${SLURM_ARRAY_TASK_ID}
a=${a_values[$(( trial % ${#a_values[@]} ))]}
trial=$(( trial / ${#a_values[@]} ))
b=${b_values[$(( trial % ${#b_values[@]} ))]}
trial=$(( trial / ${#b_values[@]} ))
c=${c_values[$(( trial % ${#c_values[@]} ))]}

python src/train_map_dynamics.py --nz $a --tau $b --lr $c  --folder_name "model/4_layer/" --num_epoch 25000