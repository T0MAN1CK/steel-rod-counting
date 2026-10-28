"""Weighted MSE: penalize errors near annotated rod centres more strongly."""

import torch


def heatmap_loss(logits, target, alpha=10.0):
    if alpha < 0:
        raise ValueError("Loss weighting must be nonnegative.")
    prediction = torch.sigmoid(logits.float())
    weights = 1 + alpha * target
    squared_error = (prediction - target).square()
    return (weights * squared_error).mean()
