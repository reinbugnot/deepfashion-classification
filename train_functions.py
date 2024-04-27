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
from utils import plot_loss_acc, get_train_valid_loader, unwrap_and_calc_loss, compute_avg_class_acc, calc_distribution, get_test_loader
from model import CustomResNet, CustomSEResNeXt, CustomSEResNeXt_v2

def train(model, dataloader, criterion_weights, optimizer, device, smoothing):
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model.forward(imgs)

        logits_converted, loss = unwrap_and_calc_loss(logits, labels, criterion_weights, num_classes = [7, 3, 3, 4, 6, 3], smoothing=smoothing)

        loss.backward()

        optimizer.step()

        acc = compute_avg_class_acc(labels, logits_converted)

        epoch_loss += loss.item()
        epoch_acc += acc

    train_loss = epoch_loss / len(dataloader)
    train_acc = epoch_acc / len(dataloader)

    return train_loss, train_acc

def evaluate(model, dataloader, criterion_weights, device):
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device) 

            logits = model.forward(imgs)

            logits_converted, loss = unwrap_and_calc_loss(logits, labels, criterion_weights, num_classes = [7, 3, 3, 4, 6, 3])
            
            acc = compute_avg_class_acc(labels, logits_converted)

            epoch_loss += loss.item()
            epoch_acc += acc

    val_loss = epoch_loss / len(dataloader)
    val_acc = epoch_acc / len(dataloader)

    return val_loss, val_acc

def test(args, model, test_loader, model_checkpoint, criterion_weights):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    with open(f'prediction.txt', 'w') as f:  # Open the file in write mode
        with torch.no_grad():
            for test_imgs, test_labels in test_loader:
                test_imgs = test_imgs.to(device)
                test_labels = test_labels.to(device) # None Array

                logits = model.forward(test_imgs)
                logits_converted, _ = unwrap_and_calc_loss(logits, test_labels, criterion_weights, num_classes = [7, 3, 3, 4, 6, 3])
                
                # Save logits_converted to a .txt file
                np.savetxt(f, logits_converted.cpu().numpy(), fmt='%d')

    return logits_converted

def training_loop(args, model, epochs = 10, print_params = True, print_info = True, print_train_progress = True, save_checkpoints = True, **kwargs):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    ## =========  Set Hyperparameters =========  ##

    # Regularization
    smoothing = kwargs.get('smoothing', args.smoothing)
    weight_decay = kwargs.get('wd', args.wd)

    # Optimization
    learning_rate = kwargs.get('lr', args.lr)
    batch_size = kwargs.get('batch_size', args.batch_size)

    if print_params:
        print("\nHYPERPARAMETERS --")
        print(f'Smoothing: {smoothing}')
        print(f'Weight Decay: {weight_decay}')
        print(f'Learning Rate: {learning_rate}')
        print(f'Batch Size: {batch_size}')

    ## ========= Define Training Objects =========  ##

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data Loader
    train_loader, valid_loader = get_train_valid_loader(args.dataset_dir, batch_size, True, save_images=args.save_images)

    # Calculate inverse class weights to handle data imbalance
    counts, criterion_weights = calc_distribution(train_loader, device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning Rate Scheduling
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=epochs)

    # Learning Rate Warm-up
    warmup_scheduler = warmup.LinearWarmup(optimizer, 5)

    ## ========= Print Info ========== ##

    if print_info:
        print('\nTRAINING INFO --')
        print(f'Device: \n{device}')
        print(f'Criterion Weights: \n{criterion_weights}')
        print(f'Model: \n{model}')
        print(f"Number of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        print('\nTRAINING PROGRESS --')

    ## ========== TRAIN LOOP ============ ##

    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    best_val_acc = float('-inf')
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss, train_acc =  train(model, train_loader, criterion_weights, optimizer, device, smoothing)
        val_loss, val_acc = evaluate(model, valid_loader, criterion_weights, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            if save_checkpoints:
                model_name_acc = f'./model_weights/acc/model_{args.fig_name[:-4]}_acc.pt'
                torch.save(model.state_dict(), model_name_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            if save_checkpoints:
                model_name_loss = f'./model_weights/loss/model_{args.fig_name[:-4]}_loss.pt'
                torch.save(model.state_dict(), model_name_loss)

        # Record Accuracy and loss
        stat_training_loss.append(train_loss)
        stat_training_acc.append(train_acc)
        stat_val_loss.append(val_loss)
        stat_val_acc.append(val_acc)

        if print_train_progress:
            print(f"Epoch {epoch} -- LR: {scheduler.get_lr()[0]:.4f} -- Training Loss: {train_loss:.3f} | Training Acc: {train_acc:.3f} | Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.3f}")
        
        # Apply LR Warmup
        with warmup_scheduler.dampening():
            if warmup_scheduler.last_step + 1 >= 5:
                scheduler.step()

    # Save Final Model
    if save_checkpoints:
        model_name_final = f'./model_weights/final/model_{args.fig_name[:-4]}_final.pt'
        torch.save(model.state_dict(), model_name_final)

    ## ========== Plot ========== ##
    plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)

    return (stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc), criterion_weights