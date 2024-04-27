#!/bin/bash 
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=12G 
#SBATCH --cpus-per-task=5
#SBATCH --job-name=DLProj-small
#SBATCH --output=results/output_%x_%j.out
#SBATCH --error=results/error_%x_%j.err

module load anaconda3/23.5.2 
eval "$(conda shell.bash hook)" 
conda activate dev-pytorch

export CUBLAS_WORKSPACE_CONFIG=:16:8

python main.py \
--dataset_dir './FashionDataset/' \
--seed 0 \
--batch_size 256 \
--epochs 60 \
--lr_scheduler \
--lr 0.009452847508848424 --wd 0.000887541785266803 \
--smoothing 0.0 \
--fig_name lr-wd-smoothing.png \
--tuning_optuna --n_trials 4 --trial_epochs 25 \
--test \

# --dropout_p 0.5 \
# --wd 0.001

## Regularizers
# lr_scheduler / wd
# dropout
# smoothing
# data augmentation