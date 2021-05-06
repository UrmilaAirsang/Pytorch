import torch
import torch.nn.functional as F
# from utils import compute_yolo_loss

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_segmentation_loss(plane_pred, target, metrics, bce_weight=0.5):
    target = target.type_as(plane_pred)
    bce = F.binary_cross_entropy_with_logits(plane_pred, target)

    plane_pred = torch.sigmoid(plane_pred)
    dice = dice_loss(plane_pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def planar_epoch_print_metrics(metrics, epoch_samples):
    outputs = []
    outputs.append(metrics['bce'] / epoch_samples)
    outputs.append(metrics['dice'] / epoch_samples)
    outputs.append(metrics['loss'] / epoch_samples)

    return outputs
