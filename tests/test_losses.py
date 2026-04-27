# tests/test_losses.py
import torch
import torch.nn.functional as F
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.losses import FocalLoss


def test_focal_equals_bce_when_gamma_zero():
    """With gamma=0 and alpha=0.5, FocalLoss reduces to standard BCE."""
    torch.manual_seed(0)
    logits  = torch.randn(64)
    targets = torch.randint(0, 2, (64,)).float()

    fl  = FocalLoss(gamma=0.0, alpha=0.5)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    assert abs(fl.item() - bce.item()) < 1e-4, (
        f"FocalLoss(gamma=0) = {fl.item():.6f}, BCE = {bce.item():.6f}"
    )


def test_focal_down_weights_easy_samples():
    """FocalLoss(gamma>0) < FocalLoss(gamma=0) when predictions are confident."""
    torch.manual_seed(1)
    # Very confident correct predictions -> easy samples
    logits  = torch.tensor([5.0, 5.0, -5.0, -5.0])
    targets = torch.tensor([1.0, 1.0,  0.0,  0.0])

    loss_g0 = FocalLoss(gamma=0.0, alpha=0.5)(logits, targets)
    loss_g2 = FocalLoss(gamma=2.0, alpha=0.5)(logits, targets)

    assert loss_g2.item() < loss_g0.item(), (
        "Focal(gamma=2) should be smaller than Focal(gamma=0) for easy samples"
    )


def test_focal_output_is_scalar():
    logits  = torch.randn(32)
    targets = torch.randint(0, 2, (32,)).float()
    loss = FocalLoss(gamma=2.0, alpha=0.5)(logits, targets)
    assert loss.shape == torch.Size([])


def test_focal_loss_is_positive():
    logits  = torch.randn(32)
    targets = torch.randint(0, 2, (32,)).float()
    loss = FocalLoss(gamma=2.0, alpha=0.5)(logits, targets)
    assert loss.item() > 0
