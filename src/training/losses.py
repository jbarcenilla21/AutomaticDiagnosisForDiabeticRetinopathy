"""Loss functions for DR binary classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Down-weights easy/well-classified examples exponentially, forcing the
    optimizer to concentrate gradient on hard boundary cases (e.g. mild DR
    with a single faint microaneurysm).

    Use with BCEWithLogitsLoss-compatible inputs: raw logits and float targets.

    Args:
        gamma: focusing parameter (default 2.0). Higher -> harder focus on
               misclassified examples. 0 reduces to standard weighted BCE.
        alpha: positive-class weight in [0, 1] (default 0.5 for balanced
               batches via WeightedRandomSampler).
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N,) raw unnormalized scores.
            targets: (N,) binary labels {0.0, 1.0}.
        """
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t     = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss    = alpha_t * (1.0 - p_t) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
