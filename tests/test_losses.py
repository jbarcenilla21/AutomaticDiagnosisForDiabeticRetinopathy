# tests/test_losses.py
import torch
import torch.nn.functional as F
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.losses import FocalLoss


def test_focal_equals_bce_when_gamma_zero():
    """With gamma=0 and alpha=0.5, FocalLoss(p) = 0.5 * BCE(p).
    alpha=0.5 applies equal weight to both classes; the 0.5 factor does not
    affect optimization since it's a constant scale on the loss landscape.
    """
    torch.manual_seed(0)
    logits  = torch.randn(64)
    targets = torch.randint(0, 2, (64,)).float()

    fl  = FocalLoss(gamma=0.0, alpha=0.5)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    assert abs(fl.item() - 0.5 * bce.item()) < 1e-4, (
        f"FocalLoss(gamma=0, alpha=0.5) = {fl.item():.6f}, 0.5*BCE = {0.5*bce.item():.6f}"
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


def test_alpha_weights_positives_more_when_alpha_high():
    """alpha > 0.5 should up-weight the positive class relative to negative."""
    # 4 negatives predicted well, 4 positives predicted well — same difficulty
    logits  = torch.tensor([-3., -3., -3., -3.,  3.,  3.,  3.,  3.])
    targets = torch.tensor([ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.])

    loss_none = FocalLoss(gamma=0.0, alpha=0.75, reduction='none')(logits, targets)
    neg_loss = loss_none[:4].mean()
    pos_loss = loss_none[4:].mean()

    assert pos_loss > neg_loss, (
        f"alpha=0.75 should weight positives more: pos={pos_loss:.4f}, neg={neg_loss:.4f}"
    )
