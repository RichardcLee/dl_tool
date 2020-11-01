# richard lee
# 2020.10.22
from torch import Tensor
import torch

# pixel-wise metric


# 准确率 accuracy
def accuracy(FP, FN, TP, TN, eps):
    return (TP + TN + eps) / (TP + TN + FP + FN + eps)


# 灵敏性/召回率/查全率 Sensitivity / recall
def recall(FP, FN, TP, TN, eps):
    return (TP + eps) / (TP + FN + eps)


# 精度/查准率 Precision
def precision(FP, FN, TP, TN, eps):
    return (TP + eps) / (TP + FP + eps)


# 特异性 specificity
def specificity(FP, FN, TP, TN, eps):
    return (TN + eps) / (FP + TN + eps)


# 交并比 IOU
def iou(FP, FN, TP, TN, eps):
    return (TP + eps) / (TP + FP + FN + eps)


# IOU by Tensor vs Tensor
def iou(output: Tensor, target: Tensor):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


# Dice coefficient / F1 score
def dice_coeff(FP, FN, TP, TN, eps):
    return (2 * TP + eps) / (2 * TP + FN + FP + eps)


# Dice coefficient / F1 score
def dice_coeff(img1: Tensor, img2: Tensor):
    smooth = 1e-5

    i_flat = img1.view(-1)
    t_flat = img2.view(-1)

    intersection = (i_flat * t_flat).sum()
    dice = (2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth)

    return dice.item()


# 表面重叠度 overlap
def overlap(FP, FN, TP, TN, eps):
    return (TP + eps) / (min(TP + FN + FP, TP + FP) + eps)


