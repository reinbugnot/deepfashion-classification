#!/bin/bash 
#SBATCH --partition=SCSEGPU_M1
#SBATCH --qos=q_amsai
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=25G 
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
--batch_size 256 \
--epochs 5 \
--lr_scheduler \
--lr 0.007 --wd 0.001 \
--dropout_p 0.5 \
--smoothing 0.001 \
--seed 0 \
--fig_name bs=256-epoch=5-lr_sch-lr_warmup-wd=0.001-lr=0.007-dropout=0.5-new_data_augments-smoothing=0.001.png \
--tuning_optuna \
#--test \
#--test_tag 'lower_smoothing' \