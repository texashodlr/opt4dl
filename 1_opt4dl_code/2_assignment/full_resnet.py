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
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import os

#plt.style.use('ggplot')

# Section XX: BasicBlock Class Definition
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        """
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        """
    def forward(self, x: Tensor) -> Tensor:
        identity = x # Stores copy of input tensor
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out += self.shortcut(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity # Similar to shortcutting but it's __residual__!
        out = self.relu(out)
        return out

# Section XX: ResNet Class Definition
class ResNet18(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_layers:int,
        block: Type[BasicBlock],
        num_classes: int = 1000
    ) -> None:
        super(ResNet18, self).__init__()
        self.expansion = 1
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    
    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        # strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(
            block(
                self.in_channels, 
                out_channels, 
                stride, 
                self.expansion, downsample))
        self.in_channels = out_channels * self.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    print(f" --- ResNet-18 successfully completed ---")

# Section XX: Training the Model
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print("--- Training the model ---")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(image)
        # Calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Back propagation
        loss.backward()
        # Update the weights
        optimizer.step()
    # Loss and acc for the completed epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc
    
# Section XX: Testing the Model
def test(model, testloader, criterion, device):
    model.eval()
    print("--- Testing the model ---")
    test_running_loss = 0.0
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # optimizer.zero_grad()
            # Forward pass
            outputs = model(image)
            # Calculate the loss
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            # Calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()
            # Back propagation
            loss.backward()
            # Update the weights
            optimizer.step()
    # Loss and acc for the completed epoch
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

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
model = ResNet18(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
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