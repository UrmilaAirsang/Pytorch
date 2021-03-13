# -*- coding: utf-8 -*-
"""S12_train_file.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iGfXxk3dzZlpUhdlyxtjlPw4M0A6-RRE
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

class customTraining():
    def __init__(self,train_loader, model, loss_func=F.nll_loss,reg=(0,0),device='cuda',):
        self.train_loader,self.model = train_loader,model
        self.train_losses,self.train_acc=[],[]
        # self.test_losses,self.test_acc=[],[]
        self.loss_func=loss_func
        self.lambda_l1,self.weight_decay=reg
        self.device=device

    def train(self, optimizer,lambda_l1=0):
        # train_losses=[],train_acc=[]
        # self.train_losses.append(train_losses)
        # self.train_acc.append(train_acc)
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct=0
        processed=0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            y_pred = self.model(data)
            loss = self.loss_func(y_pred, target)
            l1=0
            for p in self.model.parameters():
                l1 = l1 +p.abs().sum()
            loss= loss +self.lambda_l1*l1
            self.train_losses.append(loss)
            loss.backward()
            optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
        return self.train_losses,self.train_acc

    def trainWithLRStepping(self, optimizer,lr_scheduler,lambda_l1=0):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct=0
        processed=0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            y_pred = self.model(data)
            loss = self.loss_func(y_pred, target)
            l1=0
            for p in self.model.parameters():
                l1 = l1 +p.abs().sum()
            loss= loss +self.lambda_l1*l1
            self.train_losses.append(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            
            pred = y_pred.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
        return self.train_losses,self.train_acc

from tqdm import tqdm

# train_losses = []
# test_losses = []
# train_acc = []
# test_acc = []
           
def train(model, device, train_loader, optimizer, epoch,criterion):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    #train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    #train_acc.append(100*correct/processed)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

class customTrainingWithoutStepping():
    def __init__(self,train_loader, model, loss_func=F.nll_loss,reg=(0,0),device='cuda',):
        self.train_loader,self.model = train_loader,model
        self.train_losses,self.train_acc=[],[]
        # self.test_losses,self.test_acc=[],[]
        self.loss_func=loss_func
        self.lambda_l1,self.weight_decay=reg
        self.device=device

    def train(self, optimizer,lambda_l1=0):
        # train_losses=[],train_acc=[]
        # self.train_losses.append(train_losses)
        # self.train_acc.append(train_acc)
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct=0
        processed=0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            y_pred = self.model(data)
            loss = self.loss_func(y_pred, target)
            l1=0
            for p in self.model.parameters():
                l1 = l1 +p.abs().sum()
            loss= loss +self.lambda_l1*l1
            self.train_losses.append(loss)
            loss.backward()
            optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
        return self.train_losses,self.train_acc

    def trainWithoutStepping(self, optimizer,lr_scheduler,lambda_l1=0):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct=0
        processed=0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            y_pred = self.model(data)
            loss = self.loss_func(y_pred, target)
            l1=0
            for p in self.model.parameters():
                l1 = l1 +p.abs().sum()
            loss= loss +self.lambda_l1*l1
            self.train_losses.append(loss)
            loss.backward()
            optimizer.step()
        
            pred = y_pred.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
        return self.train_losses,self.train_acc