# -*- coding: utf-8 -*-
"""S7_CIFAR_DataLoadFile.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13Nm2XnybKrIbwDw_zxQ3az8jI4IoxkDf
"""

import torchvision
import torchvision.transforms as transforms


class CIFAR10DataLoader():
    def __init__(self):
        super(Net, self).__init__()
        transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.RandomRotation((-15.0, 15.0), fill=(1,)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                                 shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return trainset, trainloader, testset, testloader, classes

