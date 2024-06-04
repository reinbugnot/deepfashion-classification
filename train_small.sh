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
--beta1 0.9 \
--beta2 0.999 \
--smoothing 0.00931113078764147 \
--dropout_p 0.460510165914759 \
--lr 0.00945284750884842 --wd 0.000887541785266803 \
--fig_name lr-wd-smoothing-dropout-data_aug-batch_size-adam_betas.png \
--test \
--tuning_optuna --n_trials 10 --trial_epochs 25 \