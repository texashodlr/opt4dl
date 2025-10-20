"""
Full implementation of ResNet-18 in Python from scratch
Built from: 
1. https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/
2. https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
3. https://discuss.pytorch.org/t/how-to-add-a-l2-regularization-term-in-my-loss-function/17411
4. https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
"""
# Section 0: Imports
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Type
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import os

plot_dir = os.path.join(os.getcwd())

# Section 01: BasicBlock Class Definition
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1
    ) -> None:
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels,
                                out_channels, 
                                kernel_size=3, 
                                stride=self.stride, 
                                padding=1, 
                                bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, 
                               out_channels*self.expansion, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 
                          out_channels, 
                          kernel_size=1, 
                          stride=self.stride, 
                          bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x                            # Stores reference of input tensor
        if self.stride != 1 or self.in_channels != self.out_channels:
            identity = self.shortcut(x)         # Tensor shape resizing check
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity                         # Residual Path
        out = self.relu(out)
        return out

# Section 02: ResNet Class Definition
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
            kernel_size=3, 
            stride=1,
            padding=1,
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
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
        
    
    def forward(self, x: Tensor) -> Tensor:
                                        # Input Stem
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
                                        # Layer blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
                                        # Output 
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Section 03: Training the Model
def train(model, trainloader, optimizer, l2_lambda, criterion, device):
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

        """ SGD LOSS """
        # loss = criterion(outputs, labels) # --> Vanilla Loss with no L2 Reg.

        """ L2 Regularization with Adam"""
        # loss = criterion(outputs, labels) + l2_lambda * torch.sum(torch.stack([p.pow(2).sum() for p in model.parameters() if p.requires_grad and p.dim() > 1 ]))

        """ Part B Weight Decay with similiar Lambda """
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
    
# Section 04: Testing the Model
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
            outputs = model(image)
            # Calculate the loss
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            # Calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()
    # Loss and acc for the completed epoch
    epoch_loss = test_running_loss / counter
    epoch_acc = 100. * (test_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc

# Section 05: Early Stopping Class Definition
class EarlyStopper:
    def __init__(self, patience: int = 1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print(f"+++ A new testing loss minimum has been achieved! +++")
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"+++ A testing platuea has been hit, counter: {self.counter} +++")
            if self.counter >= self.patience:
                return True
        return False

# Section 06: Data Loading from Fashion-MNIST
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

# Section 07: Plotter
def save_plots(train_acc, test_acc, train_loss, test_loss, name=None):
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        test_acc, color='tab:red', linestyle='-', 
        label='test accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, name+'_accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        test_loss, color='tab:red', linestyle='-', 
        label='test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, name+'_loss.png'))

# Section 08: User Input
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=63)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--l2_lambda', type=float, default=0.0)
args = vars(parser.parse_args())

# Section 09: Setting training seeds
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args['seed'])
random.seed(args['seed'])

# Section 10: CUDA Device and Data Loading
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_data(batch_size=args['batch_size'])

# Section 11: Model Definition
print('--- Training ResNet-18 from scratch ---')
model = ResNet18(img_channels=1, num_layers=18, block=BasicBlock, num_classes=10).to(device)
print(model)

# Section 12: Parameter printing
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

## Optimizer ##
optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'])

""" Adam (Q3/Part: [A,B,D] """
# optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

""" AdamW (Q3/Part: [C,D]) """
# optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

## Loss Function ## 
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':

    # Training losses and accuracies
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    #local_min_delta = args['l2_lambda'] * 0.01
    local_min_delta = args['weight_decay'] * 0.01
    early_stopper = EarlyStopper(patience=2, min_delta=local_min_delta)

    for epoch in range(args['epochs']):
        print(f"--- EPOCH: {epoch+1} of {args['epochs']}")
        train_epoch_loss, train_epoch_acc = train(
            model,
            train_loader,
            optimizer,
            args['l2_lambda'],
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
        
        if early_stopper.early_stop(test_epoch_loss):
            print(f"### Test loss has failed to decrease, stopping early! ###")
            break
        else:
            print(f"### Test loss continues to decrease, not stopping early! ###")
        print('-'*50)
    save_plots(
        train_acc,
        test_acc,
        train_loss,
        test_loss,
        name="OPT4DL_Conor_Devlin_ResNet-18_from_Scratch" 
    )
    print('CONGRATS! TRAINING COMPLETE!')