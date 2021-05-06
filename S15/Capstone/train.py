import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import time

from collections import defaultdict
from tqdm import tqdm

from model import CustomNet
from utils import freeze, unfreeze
from yolo.utils import compute_yolo_loss
from test import test
from plane_segmentation.loss import calc_segmentation_loss, planar_epoch_print_metrics


def train(data_cfg, model, midas_model, train_dataloader, test_dataloader,
          start_epoch, epochs, img_size, optimizer, scheduler=None):
    mse_loss = nn.MSELoss()
    nb = len(train_dataloader)  # number of batches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    print('Starting training for %g epochs...' % epochs)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4).to(device)  # mean losses
        avg_midas_loss = 0
        print('\n-----------------Train---------------\n')
        print(('\n' + '%12s' * 12) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'yolo_total', 'targets',
                                      ' midas_loss', 'plane_bce', 'plane_dice', 'plane_total', 'img_size'))
        pbar = tqdm(enumerate(train_dataloader), total=nb)  # progress bar
        for i, (imgs, targets, seg_masks, paths, _) in pbar:
            # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            #imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            imgs = imgs.float()
            imgs = imgs.to(device)
            seg_masks = seg_masks.to(device)
            targets = targets.to(device)

            midas_target = midas_model.forward(imgs)
            # Run model
            midas_pred, plane_pred, pred = model(imgs)

            # Compute YOLO loss
            yolo_loss, yolo_loss_items = compute_yolo_loss(pred, targets, model)
            if not torch.isfinite(yolo_loss):
                print('WARNING: non-finite loss, ending training ', yolo_loss_items)

            # Compute midas loss
            midas_loss = mse_loss(midas_pred, midas_target) / 1e5

            metrics = defaultdict(float)

            # Compute Plane Segmentation loss
            plane_seg_loss = calc_segmentation_loss(plane_pred, seg_masks, metrics)
            plane_seg_print = planar_epoch_print_metrics(metrics, i + 1)

            loss = midas_loss + yolo_loss + plane_seg_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + yolo_loss_items) / (i + 1)  # update mean losses
            avg_midas_loss = (avg_midas_loss * i + midas_loss.detach()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%12s' * 2 + '%12g' * 10) % (
            '%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), avg_midas_loss, *plane_seg_print, img_size)

            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------
        if scheduler:
            scheduler.step()
        mloss = torch.zeros(4).to(device)  # mean losses
        print('\n-----------------Test---------------\n')
        print(('\n' + '%12s' * 12) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'yolo_total', 'targets',
                                      ' midas_loss', 'plane_bce', 'plane_dice', 'plane_total', 'img_size'))
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))  # progress bar
        for i, (
        imgs, targets, seg_masks, paths, _) in pbar:  # batch -------------------------------------------------------------
            model.eval()
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            seg_masks = seg_masks.to(device)
            targets = targets.to(device)

            midas_target = midas_model.forward(imgs)
            # Run model
            midas_pred, plane_pred, _, pred = model(imgs)

            # Compute YOLO loss
            yolo_loss, yolo_loss_items = compute_yolo_loss(pred, targets, model)

            # Compute midas loss
            midas_loss = mse_loss(midas_pred, midas_target) / 1e5

            metrics = defaultdict(float)

            # Compute Plane Segmentation loss
            plane_seg_loss = calc_segmentation_loss(plane_pred, seg_masks, metrics)
            plane_seg_print = planar_epoch_print_metrics(metrics, i + 1)

            optimizer.step()
            optimizer.zero_grad()
            # ema.update(model)

            # Print batch results
            mloss = (mloss * i + yolo_loss_items) / (i + 1)  # update mean losses
            avg_midas_loss = (avg_midas_loss * i + midas_loss.detach()) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%12s' * 2 + '%12g' * 10) % (
            '%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), avg_midas_loss, *plane_seg_print, img_size)

            pbar.set_description(s)
        results, maps = test(data_cfg,
                             model=model,
                             single_cls=False,
                             iou_thres=0.6,
                             dataloader=test_dataloader)

    # end epoch ----------------------------------------------------------------------------------------------------

