# -*- coding: utf-8 -*-
"""S10_test_file.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tkgiPdODLdrCAAAQBxWkmarly4-Twmb8
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision.transforms as transforms
# from torch.optim.lr_scheduler import StepLR

class customTest():
    def __init__(self,test_loader, model, loss_func=F.nll_loss, device='cuda'):
        self.test_loader,self.model = test_loader,model
        # self.train_losses,self.train_acc=[],[]
        self.test_losses,self.test_acc=[],[]
        self.loss_func=loss_func
        # self.lambda_l1,self.weight_decay=reg
        self.device=device

    def test(self):
        # ,test_losses=[],test_acc=[]
        # self.test_losses.append(test_losses)
        # self.test_acc.append(test_acc)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_func(output, target, reduction='sum').item()  
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)


        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        self.test_acc.append(100. * correct / len(self.test_loader.dataset))

        return self.test_losses,self.test_acc

import torch

def test(model, device, test_loader):
    model.eval()
    #test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
           # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    #test_loss /= len(test_loader.dataset)
    #test_losses.append(test_loss)

    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format( correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    #test_acc.append(100. * correct / len(test_loader.dataset))