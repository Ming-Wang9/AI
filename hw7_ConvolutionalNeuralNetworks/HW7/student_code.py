# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms

import math



class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        side = input_shape[0]

        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        side = math.floor((side - 5) / 1 + 1)
        self.pool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)
        side = math.floor((side - 2) / 2 + 1)

        self.conv_layer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        side = math.floor((side - 5) / 1 + 1)
        self.pool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        side = math.floor((side - 2) / 2 + 1)

        self.flatten_layer = nn.Flatten()
        self.linear_layer1 = nn.Linear(in_features=16 * side * side, out_features=256)
        self.linear_layer2 = nn.Linear(in_features=256, out_features=128)
        self.output_layer = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        shape_dict = {}
        # certain operations
        x = F.relu(self.conv_layer1(x))
        x = self.pool_layer1(x)
        shape_dict[1] = x.size()

        x = F.relu(self.conv_layer2(x))
        x = self.pool_layer2(x)
        shape_dict[2] = x.size()

        x = self.flatten_layer(x)
        shape_dict[3] = x.size()

        x = F.relu(self.linear_layer1(x))
        shape_dict[4] = x.size()

        x = F.relu(self.linear_layer2(x))
        shape_dict[5] = x.size()

        x = self.output_layer(x)
        shape_dict[6] = x.size()

        return x, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel() / 1e6
    
    model_params = total_params


    return model_params



def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
