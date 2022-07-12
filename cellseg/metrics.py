from typing import Callable, Union
from ignite.metrics import Metric
import torch
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import segmentation_models_pytorch as smp
import ignite


def binary_one_hot_output_transform(output):
    y_pred, y = output
    y_pred = torch.squeeze(y_pred, dim=1)
    y_pred = y_pred.sigmoid().round().long()
    y_pred = ignite.utils.to_onehot(y_pred, 2)
    y = y.long()
    return y_pred, y


def get_metrics(loss_fn):
    cm = ignite.metrics.ConfusionMatrix(num_classes=2, output_transform=binary_one_hot_output_transform)
    dice = ignite.metrics.DiceCoefficient(cm, ignore_index=0)
    jaccard = ignite.metrics.JaccardIndex(cm, ignore_index=0)
    iou = ignite.metrics.IoU(cm, ignore_index=0)
    miou = ignite.metrics.mIoU(cm, ignore_index=0)
    loss = ignite.metrics.Loss(loss_fn)
    metrics = {"cm": cm, "dice": dice, "jaccard": jaccard, "iou": iou, "miou": miou, "loss": loss}
    return metrics