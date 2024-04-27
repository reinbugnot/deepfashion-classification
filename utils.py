import os
import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import weibull
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from dataset import FashionNet_Dataset


# VISUALIZATION
def plot_loss_acc(train_loss, val_loss, train_acc, val_acc, fig_name):
    x = np.arange(len(train_loss))
    max_loss = max(max(train_loss), max(val_loss))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_ylim([0,max_loss+1])
    lns1 = ax1.plot(x, train_loss, color='indigo', linewidth=2, label='train_loss')
    lns2 = ax1.plot(x, val_loss, color='indigo', linestyle='dashed', linewidth=2, label='val_loss')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ax2.set_ylim([0,1])
    lns3 = ax2.plot(x, train_acc, color='darkorange', linewidth=2, label='train_acc')
    lns4 = ax2.plot(x, val_acc, color='darkorange', linestyle='dashed', linewidth=2, label='val_acc')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)

    fig.tight_layout()
    plt.title(fig_name)

    plt.savefig(os.path.join('./diagram', fig_name))

    np.savez(os.path.join('./diagram', fig_name.replace('.png ', '.npz')), train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc)

# DATALOADERS

def get_train_valid_loader(dataset_dir, batch_size, shuffle, save_images=False, mean_std = None, pin_memory=True, num_workers=2):
    
    # Transform Parameters
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]), # Resize
        transforms.CenterCrop([200, 170]), # Zoom-in
        transforms.Resize([224, 224]), # Resize Again
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Winning Augments
    # transforms.Resize([224, 224]),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.5, 1.5), shear=15),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Transform Parameters
    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(dataset_dir)

    # Load Dataset
    train_dataset = FashionNet_Dataset(dataset_dir, './FashionDataset/split/train.txt', transform=train_transform)
    valid_dataset = FashionNet_Dataset(dataset_dir, './FashionDataset/split/val.txt', transform=val_transform)

    print(f"Shape of Training Data: {train_dataset.__len__()}")
    print(f"Shape of Validation Data: {valid_dataset.__len__()}")

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True, num_workers=2)

    return train_loader, valid_loader

def get_test_loader(dataset_dir, batch_size, pin_memory=True, num_workers=2):
    
    test_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the dataset
    test_dataset = FashionNet_Dataset(dataset_dir, './FashionDataset/split/test.txt', transform=test_transform)

    print(f"Shape of Testing Data: {test_dataset.__len__()}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader

# METRICS
def unwrap_and_calc_loss(logits, labels, criterion_weights, num_classes, smoothing=0.0):
    bin_counter = 0
    loss = 0
    out = []
    for cat in range(len(num_classes)):
        # Take the argmax along the second dimension (1) for each group
        logit_slice = logits[:, bin_counter:bin_counter+num_classes[cat]]
        weight_slice = criterion_weights[bin_counter:bin_counter+num_classes[cat]]

        criterion = torch.nn.CrossEntropyLoss(weight=weight_slice, label_smoothing=smoothing).cuda()
        loss += criterion(logit_slice, labels[:, cat])

        out.append(torch.argmax(logit_slice, dim=1))
        bin_counter += num_classes[cat]
    
    return torch.stack(out, dim=1), loss/6.0

def compute_avg_class_acc(gt_labels, pred_labels):
    num_attr = 6
    num_classes = [7, 3, 3, 4, 6, 3]  # number of classes in each attribute

    per_class_acc = []
    for attr_idx in range(num_attr):
        for idx in range(num_classes[attr_idx]):
            target = gt_labels[:, attr_idx]
            pred = pred_labels[:, attr_idx]
            correct = torch.sum((target == pred) * (target == idx))
            total = torch.sum(target == idx)
            per_class_acc.append(float(correct) / float(total) if total != 0.0 else 0.0)

    return sum(per_class_acc) / len(per_class_acc)

def calc_mean_std(loader):

    # From lecture notes
    # VAR[X] = E[X**2] - E[X]**2

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    # Iterate over the entire dataset
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std

def one_hot_encode(tensor):
    # Define the maximum values for each index
    max_values = [7, 3, 3, 4, 6, 3]
    
    # Initialize an empty list to store the one-hot vectors
    one_hot_vectors = []
    
    # Iterate over the tensor and the max_values list simultaneously
    for i, max_value in enumerate(max_values):
        # Create a one-hot matrix of size (batch_size, max_value)
        one_hot_matrix = torch.zeros(tensor.shape[0], max_value)
        
        # Set the element at the index specified by value to 1
        one_hot_matrix[torch.arange(tensor.shape[0]), tensor[:, i].long()] = 1
        
        # Append the one_hot_matrix to the list
        one_hot_vectors.append(one_hot_matrix)
    
    # Concatenate the one-hot matrices to create a single matrix
    one_hot_encoded = torch.cat(one_hot_vectors, dim=1)
    
    return one_hot_encoded


def calc_distribution(loader, device):
    # Initialize a tensor to store the counts
    counts = torch.zeros(26)

    # Iterate over the batches in the dataloader
    for _, labels in loader:
        # Move the labels to the device
        labels = labels.to(device)
        
        # Apply the one_hot_encode function to the labels
        one_hot_encoded = one_hot_encode(labels)
        
        # Add the counts of 1s in the one-hot encoded labels to the counts tensor
        counts += torch.sum(one_hot_encoded, dim=0)

    # Print the counts
    print(counts)

    counts_inv = torch.pow(counts, -1)
    criterion_weights = (counts_inv * 5000.0) / 220.0

    return counts, criterion_weights




