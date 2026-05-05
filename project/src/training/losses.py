# masked L1 loss
# src/training/losses.py

import torch


def masked_l1_loss(pred, target, mask):
    diff = torch.abs(pred - target) * mask
    return diff.sum() / (mask.sum() + 1e-6)
