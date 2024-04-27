#!/bin/bash 
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=25G 
#SBATCH --cpus-per-task=10
#SBATCH --job-name=DLProj-large
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
--lr 0.00945284750884842 --wd 0.000887541785266803 \
--fig_name lr-wd-data_aug.png \
--test \
#--tuning_optuna --n_trials 4 --trial_epochs 25 \


# --dropout_p 0.5 \
# --smoothing 0.001 \
# --wd 0.001

## Regularizers
# lr_scheduler / wd
# dropout
# smoothing
# data augmentation


