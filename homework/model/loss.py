import torch
import torch.nn.functional as F


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def cross_entropy_loss(pred, target):
    return F.cross_entropy(pred, target)


def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)


def ls_loss(pred, target):
    return 0.5 * torch.mean((pred - target) ** 2)
