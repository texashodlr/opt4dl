import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Defining the NN arch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128) # FC Layer 1, Dim of weight matrix = 784x128 -> 28x28 imags
        self.fc2 = nn.Linear(128, 10)  # FC Layer 2  Dim of weight matrix = 128x10
    
    def forward(self, x):
        x = x.view(-1, 784)            # Flatten the input
        x = F.relu(self.fc1(x))        # Apply ReLU activiation function
        x = self.fc2(x)                # Output Layer
        return x

# Loading the MNIST Data
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset   = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

# Create Data Loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   # Sampling 64 data points in each iteration, 64 at a time, hence the shuffle as well
test_loader  = DataLoader(test_dataset, batch_size=1000)                # Predicting on 1000 data points then moving onto the next

# Instantiate the Model
model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()                                       # Per the notes the natural critieron is minimizing the CEL
optimizer = optim.Adam(model.parameters(), lr=0.001)                    # ADAM == advanced SGD

# Training Loop
epochs = 10
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        # Forward Pass
        output = model(data)
        # Calculate the loss
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
        # Print training progress
        if batch_idx % 100 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100. * batch_idx/len(train_loader), loss.item()))

# Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target.view_as(pred)).sum().item()

print(f"\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({100.*correct/len(test_loader.dataset)}%)\n")
