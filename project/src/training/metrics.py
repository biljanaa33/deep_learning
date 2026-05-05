# AbsRel, RMSE, delta1
# src/training/metrics.py

import torch


@torch.no_grad()
def depth_metrics(pred, target, mask):
    pred = pred[mask > 0]
    target = target[mask > 0]

    pred = torch.clamp(pred, min=1e-3)
    target = torch.clamp(target, min=1e-3)

    abs_rel = torch.mean(torch.abs(target - pred) / target)
    rmse = torch.sqrt(torch.mean((target - pred) ** 2))

    ratio = torch.maximum(target / pred, pred / target)
    delta1 = torch.mean((ratio < 1.25).float())

    return {
        "abs_rel": abs_rel.item(),
        "rmse": rmse.item(),
        "delta1": delta1.item(),
    }
