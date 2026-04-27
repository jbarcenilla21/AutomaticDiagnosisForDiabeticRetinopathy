# tests/test_transforms.py
import numpy as np
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.transforms import BenGraham, RandomCutout, PerImageNormalize, ToTensor


def _fake_uint8(h=64, w=64):
    """Synthetic uint8 fundus-like image: bright center, dark border."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 3
    for y in range(h):
        for x in range(w):
            if (y - cy) ** 2 + (x - cx) ** 2 < r ** 2:
                img[y, x] = [180, 120, 80]
    return img


def _sample(img):
    return {'image': img, 'eye': 0, 'label': 1}


def test_ben_graham_output_dtype_and_shape():
    img = _fake_uint8()
    out = BenGraham(sigma=10.0)(_sample(img))['image']
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_ben_graham_clips_to_valid_range():
    img = _fake_uint8()
    out = BenGraham(sigma=10.0)(_sample(img))['image']
    assert out.min() >= 0
    assert out.max() <= 255


def test_ben_graham_output_is_uint8():
    """BenGraham always outputs a uint8 array regardless of input content."""
    img = _fake_uint8(128, 128)
    out = BenGraham(sigma=15.0)(_sample(img))['image']
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_random_cutout_shape_unchanged():
    img = np.random.rand(64, 64, 3).astype(np.float64)
    out = RandomCutout(n_holes=2, patch_size=16, fill_value=0.5)(_sample(img))['image']
    assert out.shape == img.shape


def test_random_cutout_fills_some_pixels():
    img = np.zeros((64, 64, 3), dtype=np.float64)
    out = RandomCutout(n_holes=1, patch_size=32, fill_value=0.5)(_sample(img))['image']
    assert (out == 0.5).any(), "Expected at least one pixel filled with 0.5"


def test_per_image_normalize_zero_mean():
    img = np.random.rand(64, 64, 3).astype(np.float64)
    sample = ToTensor()(_sample(img))          # convert to (C,H,W) float32 tensor
    out = PerImageNormalize()(sample)['image']
    # Per-channel mean should be ~0 after normalization
    for c in range(3):
        assert abs(out[c].mean().item()) < 0.05, f"Channel {c} mean too far from 0"


def test_per_image_normalize_unit_std():
    img = np.random.rand(64, 64, 3).astype(np.float64)
    sample = ToTensor()(_sample(img))
    out = PerImageNormalize()(sample)['image']
    for c in range(3):
        std = out[c].std().item()
        assert 0.9 < std < 1.1, f"Channel {c} std={std:.3f}, expected ~1.0"
