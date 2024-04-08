#!/bin/bash 
#SBATCH --partition=SCSEGPU_M2
#SBATCH --qos=q_dmsai
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=25G 
#SBATCH --job-name=DLAssgn1
#SBATCH --output=results/output_%x_%j.out
#SBATCH --error=results/error_%x_%j.err

module load anaconda3/23.5.2 
eval "$(conda shell.bash hook)" 
conda activate dev-pytorch

export CUBLAS_WORKSPACE_CONFIG=:16:8

python main.py \
--dataset_dir './FashionDataset/' \
--batch_size 256 \
--epochs 1 \
--lr_scheduler \
--lr 0.007 --wd 0.001 \
--dropout_p 0.5 \
--smoothing 0.001 \
#--test \
#--test_tag 'lower_smoothing' \
--seed 0 \
--fig_name bs=256-epoch=150-lr_sch-lr_warmup-wd=0.001-lr=0.007-dropout=0.5-new_data_augments-smoothing=0.001.png \

# JOBS
# 22267
# 22369 bs=128-epoch=100-lr_sch-lr_warmup-wd=0.00095-lr=0.005-dropout=0.5
# 22394 bs=128-epoch=100-lr_sch-lr_warmup-wd=0.00095-lr=0.005-dropout=0.5-data_augments
# 22449/22477 bs=128-epoch=80-lr_sch-lr_warmup-wd=0.0008-lr=0.005-dropout=0.5-data_augments (lighter augments) -- TEST ACC: 64.4%
# 22510 (stronger augments)
# 22553 bs=128-epoch=120-lr_sch-lr_warmup-wd=0.0008-lr=0.005-dropout=0.5-data_augments-smoothing=0.01.png -- TEST ACC: 70.0018% TOP 9
# 22615 bs=128-epoch=120-lr_sch-lr_warmup-wd=0.0008-lr=0.005-dropout=0.5-data_augments-smoothing=0.015.png
# 22801 bs=256-epoch=150-lr_sch-lr_warmup-wd=0.0008-lr=0.005-dropout=0.5-data_augments-smoothing=0.015
# 22856 bs=128-epoch=150-lr_sch-lr_warmup-wd=0.0008-lr=0.005-dropout=0.5-data_augments-smoothing=0.009.png
# 22910 replica of top 9 score
# 22982 bs=256-epoch=120-lr_sch-lr_warmup-wd=0.0008-lr=0.005-dropout=0.5-data_augments-smoothing=0.01-2.png
# 23042 bs=128-epoch=120-lr_sch-lr_warmup-wd=0.0008-lr=0.005-dropout=0.5-data_augments-smoothing=0.01-2.png
# 23373 smoothing=0.001