# -*- coding: utf-8 -*-
"""S9_CIFAR_DataLoader.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wvIRgNZypIXkrZYffatB8e2bEkEgjmII
"""

import torchvision
import torchvision.transforms as transforms
import torch

def CIFAR10DataLoader(input_batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=input_batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=input_batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes

def CIFAR10Check_Mean_STD(input_batch_size):
  transform = transforms.Compose(
      [transforms.ToTensor()])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=input_batch_size,
                                            shuffle=True, num_workers=2)
  mean = 0.0
  std = 0.0
  nb_samples = 0.
  for data, C in trainloader:
    #data = images
    #print(data)
    batch_samples = data.size(0)
    # print(batch_samples)
    # print(data.size(1))
    # print(data.size(2))  
    data = data.view(batch_samples, data.size(1), -1) # data dimension = batchsize * number of channels * (h*w) = 128*3*(32*32) 
    # print(data.size(2))
    # print(type(data))
    # m = data.mean(2).sum(0)
    # print(m)
    # break
    mean += data.mean(2).sum(0) # summing for each image in a batch which gives 128*1024 values
    # mean(2) : mean of the 1024 values corresponds to the 1 channel of the image, sum(0): sum of whole batch images 
    std += data.std(2).sum(0)
    nb_samples += batch_samples

  # print('mean', mean)
  # print('std', std)
  # print('nb_samples', nb_samples)
  mean /= nb_samples
  std /= nb_samples
  return mean, std

def CIFAR10DataLoaderWithRotate(input_batch_size):
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=input_batch_size,
                                              shuffle=True, num_workers=2)

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=input_batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes

def CIFAR10DataLoaderWithRotateCrop(input_batch_size):
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.CenterCrop(30),
         transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=input_batch_size,
                                              shuffle=True, num_workers=2)

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=input_batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes

def CIFAR10DataLoaderWithRotateNormalization(input_batch_size, mean, std):
    #print(tuple(mean), tuple(std))
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
         transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=input_batch_size,
                                              shuffle=True, num_workers=2)

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=input_batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes

def CIFAR10DataLoaderWithRotateNormalizationwithConstantValues(input_batch_size):
    #print(tuple(mean), tuple(std))
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=input_batch_size,
                                              shuffle=True, num_workers=2)

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=input_batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
#from torchvision import datasets


class AlbumentationsDataset(Dataset):
  def __init__(self):
        pass
  def build_transforms(self,  train_tfms_list=[], test_tfms_list=[]):
        train_tfms_list.extend([A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]), ToTensorV2()])
        test_tfms_list.extend([A.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]), ToTensorV2()])
        return A.Compose(train_tfms_list), A.Compose(test_tfms_list)

  def get_augmentation(self,transforms):
            return lambda img: transforms(image=np.array(img))['image']

  def cifar10dataWithAugmentation(self,input_batch_size,cust_train_tfms_list=[], cust_test_tfms_list=[]):
    #Aclass = AlbumentationsDataset()
    # custom_train_tfms = [ A.RandomCrop(32, 32, p=0.8),
    #                     A.HorizontalFlip()]
    train_transform1,  test_transform1 = self.build_transforms(train_tfms_list=cust_train_tfms_list, test_tfms_list=cust_test_tfms_list)

    #print(train_transform1)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = self.get_augmentation(train_transform1))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=input_batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=self.get_augmentation(test_transform1))
    testloader = torch.utils.data.DataLoader(testset, batch_size=input_batch_size,
                                            shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes