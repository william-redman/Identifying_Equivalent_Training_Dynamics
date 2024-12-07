#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:51:20 2020

@author: wtredman
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import scipy.io
import os
import time
import numpy as np
import matplotlib.pyplot as plt

h = 5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        alpha = 1.0
        self.fc1 = nn.Linear(28*28, h, bias = False)
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight * alpha)
        self.fc2 = nn.Linear(h, 10, bias = False)
        self.fc2.weight = torch.nn.Parameter(self.fc2.weight * alpha)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.gelu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, datapath):
    model.train()
    loss_vec = np.zeros(int(np.ceil(60000. / (args.batch_size * args.log_interval))))
    weights = np.zeros([int(np.ceil(60000. / (args.batch_size * args.log_interval))), 10 * h])
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        if not datapath:
            loss.backward()
            optimizer.step()
            
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            loss_vec[counter] = loss
            weights[counter, :] = model.fc2.weight.flatten().cpu().detach().numpy()
            counter = counter + 1
            
            if args.dry_run:
                break
    return loss_vec, weights

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss

def main(id, batch_seed, init_seed, save_flag, load_flag):
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default= 60, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)') #'lr = 0.3 SGD
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Net().to(device)
    
    torch.manual_seed(init_seed)
    
    if save_flag:
        torch.save(model.state_dict(), folder_name + experiment_name + str(batch_seed) + '_initialization.pt')
    if load_flag:
        initialization = torch.load(folder_name + experiment_name + str(batch_seed) + '_initialization.pt')
        initialization['fc1.weight'] += (0.001 * (2 * torch.rand(initialization['fc1.weight'].size()) - 1) * initialization['fc1.weight'])
        initialization['fc2.weight'] += (0.001 * (2 * torch.rand(initialization['fc2.weight'].size()) - 1) * initialization['fc2.weight'])
        model.load_state_dict(initialization)
         
    data_path = ''
    save_path = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Github/Fully connected neural networks MNIST/Results/'

    os.chdir(save_path)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    Loss = torch.zeros(args.epochs)
    Train_Loss = []
    
    torch.manual_seed(batch_seed)
    
    for epoch in range(args.epochs):
        loss_vec, W = train(args, model, device, train_loader, optimizer, epoch, data_path)
        Train_Loss.append(loss_vec)
        Loss[epoch] = test(model, device, test_loader)

    Loss = Loss.data.numpy()
    
    return args, Loss, Train_Loss, W
        
if __name__ == '__main__':
    folder_name = '/Users/willredman/Documents/AIMdyn/Identifying Equivalent Optimization Algorithms/Github/Fully connected neural networks/Results/'
    experiment_name = 'SGD_MNIST_h=' + str(h) + '_relu_'
    
    random_seeds = np.array([2, 3, 6, 7, 11, 16, 17, 22, 23, 31, 33, 34, 40, 41, 42, 51, 56, 57, 61, 68, 69, 71, 87, 88, 95]) #[2, 3, 6, 7, 11, 16, 17, 22, 23, 31, 33, 34, 40, 41, 42, 51, 56, 57, 61, 68, 69, 71, 87, 88, 95])
    n_seeds = len(random_seeds)
    n_inits = 10
    for id in range(n_seeds):
        for init in range(n_inits):
            if init == 0:
                Args, Loss, Train_Loss, W = main(id, batch_seed = random_seeds[id], init_seed = init, save_flag = True, load_flag = False)
            else: 
                Args, Loss, Train_Loss, W = main(id, batch_seed = random_seeds[id], init_seed = init, save_flag = False, load_flag = True)
            np.save(folder_name + experiment_name + '_seed' + str(random_seeds[id]) + '_initialization' + str(init) + '_weights.npy', W)
            np.save(folder_name + experiment_name + '_seed' + str(random_seeds[id]) + '_initialization' + str(init) + '_Train_Loss.npy', Train_Loss)
            np.save(folder_name + experiment_name + '_seed' + str(random_seeds[id]) + '_initialization' + str(init) + '_Loss.npy', Loss)
            np.save(folder_name + experiment_name + '_seed' + str(random_seeds[id]) + '_initialization' + str(init) + '_Args.npy', Args)
    
    
    
    
    
    
    
    