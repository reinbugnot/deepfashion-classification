from pkgutil import get_data
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import pytorch_warmup as warmup
import torch.nn as nn
import os
import math

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

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data Loader
    train_loader, valid_loader = get_train_valid_loader(args.dataset_dir, args.batch_size, True, save_images=args.save_images)

    # Test Data
    if args.test:
        test_loader = get_test_loader(args.dataset_dir, args.batch_size)

    # Calc class weights to handle data imbalance
    counts = calc_distribution(train_loader, device)
    counts_inv = torch.pow(counts, -1)
    criterion_weights = (counts_inv * 5000.0) / 220.0
    print(criterion_weights)

    # MODEL
    model = CustomSEResNeXt_v2(dropout_p=args.dropout_p)
    print(model)
    model.cuda()

    print(f"Number of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Criterion
    #criterion = torch.nn.CrossEntropyLoss().cuda()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)

    # LR Warm-up
    warmup_scheduler = warmup.LinearWarmup(optimizer, 5)

    # Smoothing
    smoothing = args.smoothing

    ## ========== TRAIN ============
    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    best_val_acc = float('-inf')
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss, train_acc =  train(model, train_loader, criterion_weights, optimizer, device, smoothing)
        val_loss, val_acc = evaluate(model, valid_loader, criterion_weights, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name_acc = f'./model_weights/acc/model_{args.fig_name}_acc.pt'
            torch.save(model.state_dict(), model_name_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name_loss = f'./model_weights/loss/model_{args.fig_name}_loss.pt'
            torch.save(model.state_dict(), model_name_loss)

        # Record loss
        stat_training_loss.append(train_loss)
        stat_training_acc.append(train_acc)
        stat_val_loss.append(val_loss)
        stat_val_acc.append(val_acc)

        print(f"Epoch {epoch} -- LR: {scheduler.get_lr()[0]:.4f} -- Training Loss: {train_loss:.3f} | Training Acc: {train_acc:.3f} | Validation Loss: {val_loss:.3f} | Validation Acc: {val_acc:.3f}")
        
        with warmup_scheduler.dampening():
            if warmup_scheduler.last_step + 1 >= 5:
                scheduler.step()

    # Save Final Model
    model_name_final = f'./model_weights/final/model_{args.fig_name}_final.pt'
    torch.save(model.state_dict(), model_name_final)

    # Plot
    plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)


    # ============ MODEL INFERENCE ============

    if args.test:

        for tag, instance in [('final', model_name_final), ('acc', model_name_acc)]:

            model.load_state_dict(torch.load(instance))

            model.eval()
            with open(f'prediction_{tag}_{args.test_tag}.txt', 'w') as f:  # Open the file in write mode
                with torch.no_grad():
                    for test_imgs, test_labels in test_loader:
                        test_imgs = test_imgs.to(device)
                        test_labels = test_labels.to(device) # None Array

                        logits = model.forward(test_imgs)
                        logits_converted, _ = unwrap_and_calc_loss(logits, test_labels, criterion_weights, num_classes = [7, 3, 3, 4, 6, 3])
                        
                        # Save logits_converted to a .txt file
                        np.savetxt(f, logits_converted.cpu().numpy(), fmt='%d')

            print(f"Test Predictions generated at prediction_{tag}_{args.test_tag}.txt")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--test_tag',type=str, default='0', help='')
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd',type=float, help='')
    parser.add_argument('--beta1', type=float, default=0.9, help='')
    parser.add_argument('--beta2', type=float, default=0.999, help='')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='')
    parser.add_argument('--dropout_p', type=float, default=0.25, help='')
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
    args = parser.parse_args()
    print(args)
    main(args)
