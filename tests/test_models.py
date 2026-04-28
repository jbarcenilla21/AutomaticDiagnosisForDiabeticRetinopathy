# tests/test_models.py
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.custom_net import SEResNet9


def test_seresnet9_output_shape_small_input():
    """Forward pass with a tiny input to keep test fast."""
    model = SEResNet9()
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"


def test_seresnet9_output_shape_512():
    model = SEResNet9()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_seresnet9_parameter_count():
    model = SEResNet9()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1_500_000 < n < 3_000_000, f"Unexpected param count: {n:,}"


def test_seresnet9_output_is_logit_not_probability():
    """Output should be an unbounded logit, not a sigmoid probability."""
    torch.manual_seed(42)
    model = SEResNet9()
    model.eval()
    x = torch.randn(8, 3, 64, 64)
    with torch.no_grad():
        out = model(x).flatten()
    # At least some logits should be outside [0, 1]
    outside_unit = ((out < 0) | (out > 1)).any()
    assert outside_unit, "All outputs in [0,1] — looks like sigmoid was applied internally"


def test_seresnet9_gradients_flow():
    """Backward pass should not produce NaN gradients."""
    model = SEResNet9()
    x = torch.randn(2, 3, 64, 64)
    loss = model(x).sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
