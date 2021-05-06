from yolo.utils import parse_data_cfg

data = parse_data_cfg(r"G:\EVA5\ToGit\custom.data")
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from model import CustomNet
from utils import freeze, unfreeze
from yolo.utils import compute_yolo_loss
from test import test

from midas.midas_net import MidasNet
from datasets import *
from train import train

from torchvision import models


if __name__ == '__main__':
    hyp = {'giou': 3.54,  # giou loss gain
           'cls': 37.4,  # cls loss gain
           'cls_pw': 1.0,  # cls BCELoss positive_weight
           'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
           'obj_pw': 1.0,  # obj BCELoss positive_weight
           'iou_t': 0.225,  # iou training threshold
           'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
           'lrf': 0.0005,  # final learning rate (with cos scheduler)
           'momentum': 0.937,  # SGD momentum
           'weight_decay': 0.000484,  # optimizer weight decay
           'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
           'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
           'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
           'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
           'degrees': 1.98 * 0,  # image rotation (+/- deg)
           'translate': 0.05 * 0,  # image translation (+/- fraction)
           'scale': 0.05 * 0,  # image scale (+/- gain)
           'shear': 0.641 * 0}  # image shear (+/- deg)

    yolo_cfg = {'type': 'yolo', 'mask': [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                'anchors': np.array([[10., 13.],
                                     [16., 30.],
                                     [33., 23.],
                                     [30., 61.],
                                     [62., 45.],
                                     [59., 119.],
                                     [116., 90.],
                                     [156., 198.],
                                     [373., 326.]]), 'classes': 4, 'num': 9, 'jitter': '.3', 'ignore_thresh': '.7',
                'truth_thresh': 1, 'random': 1, 'stride': [32, 16, 8]}

    plane_segmentation_cfg = {
        "meta_data_path": "G:/EVA5/ToGit/Planercnn/content/planercnn/test/inference/"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)


    midas_model = MidasNet(r"G:\EVA5\ToGit\model-f6b98070.pt",
                           non_negative=True)
    midas_model.eval()
    midas_model.to(device)
    #print(midas_model)
    print("Model Loaded")

    # model = CustomNet("model-f46da743.pt", non_negative=True, yolo_cfg=yolo_cfg)
    model = CustomNet("G:\EVA5\ToGit\yolov3-spp-ultralytics.pt",
                      non_negative=True, yolo_cfg=yolo_cfg)

    model.gr = 1.0
    model.hyp = hyp
    model.to(device)

    #print(model)

    # freeze(model, base=True)

    # Training on images of size 64



    batch_size = 256
    img_size = 64
    test_batch_size = 256

    dataset = LoadImagesAndLabels(data['train'], plane_segmentation_cfg, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  cache_images=False,
                                  single_cls=False,
                                  seg_data=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=True,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(data['valid'], plane_segmentation_cfg,
                                                                 img_size, test_batch_size,
                                                                 hyp=hyp,
                                                                 #  rect=True,
                                                                 cache_images=True,
                                                                 seg_data=True,
                                                                 single_cls=False),
                                             batch_size=test_batch_size,
                                             num_workers=4,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=hyp['weight_decay'])

    train(data, model, midas_model, train_dataloader=dataloader, test_dataloader=testloader,
          start_epoch=0, epochs=50, img_size=64, optimizer=optimizer)

    ###################################################################################################################
    print('Img size = 128')
    batch_size = 256
    img_size = 128
    test_batch_size = 256

    dataset = LoadImagesAndLabels(data['train'], plane_segmentation_cfg, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  cache_images=False,
                                  single_cls=False,
                                  seg_data=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=True,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(data['valid'], plane_segmentation_cfg,
                                                                 img_size, test_batch_size,
                                                                 hyp=hyp,
                                                                 #  rect=True,
                                                                 cache_images=True,
                                                                 seg_data=True,
                                                                 single_cls=False),
                                             batch_size=test_batch_size,
                                             num_workers=4,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=hyp['weight_decay'])

    train(data, model, midas_model, train_dataloader=dataloader, test_dataloader=testloader,
          start_epoch=0, epochs=50, img_size=64, optimizer=optimizer)

    ###################################################################################################################
    print('Img size = 256')
    batch_size = 256
    img_size = 256
    test_batch_size = 256

    dataset = LoadImagesAndLabels(data['train'], plane_segmentation_cfg, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=False,  # rectangular training
                                  cache_images=False,
                                  single_cls=False,
                                  seg_data=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             shuffle=True,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(data['valid'], plane_segmentation_cfg,
                                                                 img_size, test_batch_size,
                                                                 hyp=hyp,
                                                                 #  rect=True,
                                                                 cache_images=True,
                                                                 seg_data=True,
                                                                 single_cls=False),
                                             batch_size=test_batch_size,
                                             num_workers=4,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=hyp['weight_decay'])

    train(data, model, midas_model, train_dataloader=dataloader, test_dataloader=testloader,
          start_epoch=0, epochs=50, img_size=64, optimizer=optimizer)
