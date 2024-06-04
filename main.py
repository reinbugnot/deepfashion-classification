import numpy as np
import torch.nn as nn
import os
import math
from pkgutil import get_data

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_warmup as warmup
import optuna

# IMPORT MODULES
from utils import plot_loss_acc, get_train_valid_loader, unwrap_and_calc_loss, compute_avg_class_acc, calc_distribution, get_test_loader
from model import CustomResNet, CustomSEResNeXt, CustomSEResNeXt_v2
from train_functions import train, evaluate, test, training_loop

# Optuna Tuning Objective
def objective(trial, args):

    # Define parameter trial space
    # Optuna will heuristically select a value from within this range to test for every trial
    # options: trial.suggest_float(), trial.suggest_int(), trial.suggest_categorical(), etc.
    params = {
        'batch_size': trial.suggest_int('batch_size', 64, 512),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2),
        'wd': trial.suggest_float('wd', 1e-4, 1e-2),
        'smoothing': trial.suggest_float('smoothing', 1e-4, 1e-2),
        'dropout_p': trial.suggest_float('dropout_p', 0.15, 0.6),
        'beta1': trial.suggest_float('beta1', 0.87, 0.93),
        'beta2': trial.suggest_float('beta2', 0.99, 0.99999999)
    }

    # Define Model
    # model = CustomSEResNeXt_v2(dropout_p=args.dropout_p)
    # model.cuda()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Training Loop
    (stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc), _, _ = training_loop(args, 
                                                                                            epochs = args.trial_epochs,
                                                                                            print_info = False, 
                                                                                            save_checkpoints = False,
                                                                                            **params)

    return np.mean(stat_val_loss[-5:])

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Call Optuna
    if args.tuning_optuna:
        study = optuna.create_study(study_name='regularization', direction="minimize", sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

        print(study.best_params)
        best_params = study.best_params
    else:
        # Manually input best_params here if not tuning
        best_params = {'lr':args.lr}
    # End of Optuna

    # Define Model
    # model = CustomSEResNeXt_v2(dropout_p=args.dropout_p)
    # model.cuda()

    # Training Loop
    (stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc), criterion_weights, model = training_loop(args, 
                                                                                                                   epochs = args.epochs,
                                                                                                                   print_params = True,
                                                                                                                   print_info = True,
                                                                                                                   print_train_progress = True, 
                                                                                                                   save_checkpoints = True, 
                                                                                                                   **best_params)

    # Test Data
    if args.test:
        test_loader = get_test_loader(args.dataset_dir, args.batch_size)
        test_loss, test_acc = evaluate(model, test_loader, criterion_weights, device)
        print(f"\n~ Inference Results ~\n Test Loss: {test_loss} | Test Acc: {test_acc}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--test_tag',type=str, default='0', help='')
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--n_trials',type=int, default=1, help='')
    parser.add_argument('--trial_epochs',type=int, default=10, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd',type=float, help='')
    parser.add_argument('--beta1', type=float, default=0.9, help='')
    parser.add_argument('--beta2', type=float, default=0.999, help='')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='')
    parser.add_argument('--dropout_p', type=float, default=0.0, help='')
    parser.add_argument('--smoothing', type=float, default=0.0, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.set_defaults(lr_scheduler=False)
    parser.add_argument('--mixup', action='store_true')
    parser.set_defaults(mixup=False)
    parser.add_argument('--alpha', type=float, default=0.2, help='')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--tuning_optuna', action='store_true')
    parser.set_defaults(tuning_optuna=False)
    args = parser.parse_args()
    print(args)
    main(args)
