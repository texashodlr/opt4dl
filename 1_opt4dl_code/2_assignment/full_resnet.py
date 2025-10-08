"""
Full implementation of ResNet-18 in Python from scratch
"""
# Section 0: Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing import Type
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import os

#plt.style.use('ggplot')


# Section XX: Data Loading from Fashion-MNIST
def get_data (batch_size=64):
    # Fashion-MNIST Data Set
    # https://github.com/zalandoresearch/fashion-mnist
    # https://docs.pytorch.org/vision/0.19/generated/torchvision.datasets.FashionMNIST.html
    dataset_train = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    dataset_test = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # Create Data Loaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader

# Section XX: Plotter
def save_plots(train_acc, test_acc, train_loss, test_loss, name=None):
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='test accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('outputs', name+'_loss.png'))

# Section XX: User Input
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=63)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = vars(parser.parse_args())

# Section XX: Setting training seeds
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(args['seed'])
random.seed(args['seed'])

# Section XX: CUDA Device and Data Loading
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_data(batch_size=args['batch_size'])

# Section XX: Model Definition
print('--- Training ResNet-18 from scratch ---')
model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
print(model)

# Section XX: Parameter printing
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

## Optimizer
optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'])

## Loss Function
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':

    # Training losses and accuracies
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []

    for epoch in range(args['epochs']):
        print(f"--- EPOCH: {epoch+1} of {args['epochs']}")
        train_epoch_loss, train_epoch_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )
        test_epoch_loss, test_epoch_acc = test(
            model,
            test_loader,
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        test_loss.append(test_epoch_loss)
        train_acc.append(train_epoch_acc)
        test_acc.append(test_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Testing loss: {test_epoch_loss:.3f}, test acc: {test_epoch_acc:.3f}")
        print('-'*50)
    save_plots(
        train_acc,
        test_acc,
        train_loss,
        test_loss,
        name="OPT4DL: Conor Devlin ResNet-18 from Scratch" 
    )
    print('CONGRATS! TRAINING COMPLETE!')