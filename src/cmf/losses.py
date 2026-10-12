"""Configurable classification losses for trial-level DEAP benchmarks.

Use raw logits (not softmax probabilities). Class weights, if supplied, are
applied per target and the weighted sample losses are normalized by the sum
of target weights to keep the scale comparable across batch compositions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def classification_loss(logits, targets, kind="ce", class_weights=None,
                        smoothing=0.1, gamma=2.0):
    """CE, label-smoothed CE, or focal CE; all return scalar differentiable loss."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Expected [batch, num_classes] logits")
    targets = targets.long().reshape(-1)
    if targets.shape[0] != logits.shape[0]:
        raise ValueError("Target and logits batch sizes differ")
    if kind not in {"ce", "label_smoothing", "focal"}:
        raise ValueError(f"Unknown loss: {kind}")
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0,1)")
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    if kind == "label_smoothing":
        sample_loss = (1.0 - smoothing) * nll - smoothing * log_probs.mean(dim=1)
    elif kind == "focal":
        pt = torch.exp(-nll)
        sample_loss = (1.0 - pt).pow(gamma) * nll
    else:
        sample_loss = nll
    if class_weights is None:
        return sample_loss.mean()
    weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    if weights.ndim != 1 or weights.numel() != logits.shape[1]:
        raise ValueError("class_weights must have one value per class")
    sample_weights = weights[targets]
    return (sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
