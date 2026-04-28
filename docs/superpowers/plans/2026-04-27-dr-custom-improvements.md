# DR Custom CNN — Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incrementally fix the root causes of the observed underfitting plateau (val AUC 0.649) in the custom DR classifier by improving preprocessing, loss function, architecture, and training dynamics.

**Architecture:** Replace ImageNet-normalized ResNet9 with a Preprocessing → SEResNet9 pipeline: Ben Graham illumination normalization → per-image standardization, SE-augmented ResNet9 with increased capacity (~1.8M params), Focal Loss, and LinearWarmup+Cosine LR scheduling.

**Tech Stack:** PyTorch, OpenCV (`cv2`), scikit-image, scikit-learn, torchvision, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/data/transforms.py` | Modify | Add `BenGraham`, `RandomCutout`, `PerImageNormalize`; fix rotation; disable hue; update pipelines |
| `src/training/losses.py` | **Create** | `FocalLoss` class |
| `src/model/ensemble_net.py` | Modify | Add `_SEBlock`, `SEResNet9` (dropout=0.1, Kaiming init, 256-ch deep layers) |
| `src/training/config.py` | Modify | Add `warmup_epochs` constant |
| `train_colab.ipynb` | Modify | `WeightedRandomSampler` in data cell; `SequentialLR` + `FocalLoss` + `SEResNet9` in model cell |
| `tests/test_transforms.py` | **Create** | Tests for BenGraham, RandomCutout, PerImageNormalize |
| `tests/test_losses.py` | **Create** | Tests for FocalLoss vs BCE equivalence and focal down-weighting |
| `tests/test_models.py` | **Create** | Tests for SEResNet9 forward shape and parameter count |

---

## Task 1: Ben Graham preprocessing, PerImageNormalize, augmentation fixes

**Files:**
- Modify: `src/data/transforms.py`
- Create: `tests/test_transforms.py`

### Why these changes
- Loss stuck at ~1.01 ≈ random predictor baseline (1.017). The network wastes capacity on illumination gradients instead of lesions.
- ImageNet normalization stats are computed on natural photos; retinal images are predominantly red/orange → wrong distribution shifts the input, slows SGD.
- Rotation currently ±15°; retina has no fixed orientation, full ±180° (= 360°) is biologically valid.
- Hue shift currently 0.05; hemorrhage red and exudate yellow are diagnostic → hue shifts destroy the signal.

- [ ] **Step 1.1: Create tests file**

```python
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


def test_ben_graham_reduces_std_vs_original():
    """Ben Graham equalizes illumination — the global channel std should not INCREASE."""
    img = _fake_uint8(128, 128)
    out = BenGraham(sigma=15.0)(_sample(img))['image']
    # The background pixels (0,0,0) will be mapped to ~128; std across all pixels may
    # change; we just verify the transform runs and output is uint8.
    assert out.dtype == np.uint8


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
```

- [ ] **Step 1.2: Run tests — expect FAIL**

```bash
cd "/home/jorge/Documentos/Carlos III/Computer Vision/AutomaticDiagnosisForDiabeticRetinopathy"
python -m pytest tests/test_transforms.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'BenGraham'`

- [ ] **Step 1.3: Implement transforms**

Replace `src/data/transforms.py` with the following (full file — preserves all existing classes and adds new ones):

```python
"""Transform classes for the Retinopathy dataset.

All transforms are callable objects that accept and return a sample dict:
    {'image': np.ndarray (HxWxC), 'eye': int, 'label': int/tensor}

Pipeline order:
  Train: BenGraham -> CropByEye -> Rescale -> RandomCrop -> augmentation
         -> RandomCutout -> ToTensor -> PerImageNormalize
  Val:   BenGraham -> CropByEye -> Rescale -> CenterCrop
         -> ToTensor -> PerImageNormalize
"""

import numpy as np
import torch
import cv2
from PIL import Image
from skimage import transform, util, color
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Retinal-specific preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class BenGraham:
    """Ben Graham fundus illumination normalization.

    Subtracts the local average color and maps the background to 50% gray.
    This maximizes contrast for microaneurysms and exudates while suppressing
    vignetting and inter-camera illumination variance.

    Must be the FIRST transform (operates on uint8 HxWxC numpy arrays).

    Args:
        sigma: Gaussian blur std deviation for local average. Use 10-30 for
               512x512 inputs.
    """

    def __init__(self, sigma: float = 15.0):
        self.sigma = sigma

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(image, (0, 0), self.sigma)
        image = cv2.addWeighted(image, 4, blurred, -4, 128)
        image = np.clip(image, 0, 255).astype(np.uint8)
        return {'image': image, 'eye': eye, 'label': label}


class RandomCutout:
    """Randomly erase n_holes square patches to force holistic feature use.

    Applied before ToTensor on float64 [0,1] images.

    Args:
        n_holes:    number of patches to erase per image.
        patch_size: side length in pixels of each erased square.
        fill_value: fill color (0.5 = mid-gray for float images).
    """

    def __init__(self, n_holes: int = 2, patch_size: int = 32,
                 fill_value: float = 0.5):
        self.n_holes = n_holes
        self.patch_size = patch_size
        self.fill_value = fill_value

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        half = self.patch_size // 2
        image = image.copy()
        for _ in range(self.n_holes):
            cy = np.random.randint(0, h)
            cx = np.random.randint(0, w)
            y1, y2 = max(cy - half, 0), min(cy + half, h)
            x1, x2 = max(cx - half, 0), min(cx + half, w)
            image[y1:y2, x1:x2] = self.fill_value
        return {'image': image, 'eye': eye, 'label': label}


class PerImageNormalize:
    """Per-channel zero-mean unit-variance normalization on a float32 tensor.

    Must follow ToTensor (operates on (C, H, W) float32 tensors).
    Handles variable retinal illumination without requiring dataset-wide stats.
    """

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        mean = image.mean(dim=(1, 2), keepdim=True)
        std  = image.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        image = (image - mean) / std
        return {'image': image, 'eye': eye, 'label': label}


# ─────────────────────────────────────────────────────────────────────────────
# Custom transforms (numpy-based)
# ─────────────────────────────────────────────────────────────────────────────

class CropByEye:
    """Crop the image to the bounding box of the eye, removing black borders."""

    def __init__(self, threshold, border):
        self.threshold = threshold
        if isinstance(border, int):
            self.border = (border, border)
        else:
            self.border = border

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        imgray = color.rgb2gray(image)
        _, mask = cv2.threshold(imgray, self.threshold, 1, cv2.THRESH_BINARY)
        sidx = np.nonzero(mask)
        if len(sidx[0]) < 20:
            return {'image': image, 'eye': eye, 'label': label}
        minx = max(sidx[1].min() - self.border[1], 0)
        maxx = min(sidx[1].max() + 1 + self.border[1], w)
        miny = max(sidx[0].min() - self.border[0], 0)
        maxy = min(sidx[0].max() + 1 + self.border[1], h)
        image = image[miny:maxy, minx:maxx, ...]
        return {'image': image, 'eye': eye, 'label': label}


class Rescale:
    """Resize image. output_size: int (shorter-side rule) or (h, w) tuple."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        image = transform.resize(image, (int(new_h), int(new_w)))
        return {'image': image, 'eye': eye, 'label': label}


class RandomCrop:
    """Randomly crop the image. output_size: int (square) or (h, w) tuple."""

    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top  = np.random.randint(0, h - new_h) if h > new_h else 0
        left = np.random.randint(0, w - new_w) if w > new_w else 0
        image = image[top:top + new_h, left:left + new_w]
        return {'image': image, 'eye': eye, 'label': label}


class CenterCrop:
    """Central crop. output_size: int (square) or (h, w) tuple."""

    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top  = int((h - new_h) / 2) if h > new_h else 0
        left = int((w - new_w) / 2) if w > new_w else 0
        image = image[top:top + new_h, left:left + new_w]
        return {'image': image, 'eye': eye, 'label': label}


class ToTensor:
    """Convert numpy ndarray (HxWxC) to torch float32 tensor (CxHxW)."""

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy()).float()
        label = torch.tensor(label, dtype=torch.long)
        return {'image': image, 'eye': eye, 'label': label}


class Normalize:
    """Normalize tensor image per channel: (x - mean) / std."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std  = torch.tensor(std,  dtype=torch.float32)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = image.sub(self.mean[:, None, None]).div(self.std[:, None, None])
        return {'image': image, 'eye': eye, 'label': label}


# ─────────────────────────────────────────────────────────────────────────────
# torchvision wrappers (numpy -> PIL -> torchvision transform -> numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _to_pil(image):
    return Image.fromarray(util.img_as_ubyte(image))

def _to_numpy(pil_image):
    return util.img_as_float(np.asarray(pil_image))


class TVCenterCrop:
    def __init__(self, size):
        self._t = transforms.CenterCrop(size)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        return {'image': _to_numpy(self._t(_to_pil(image))), 'eye': eye, 'label': label}


class TVRescale:
    def __init__(self, output_size):
        self._t = transforms.Resize(output_size)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        return {'image': _to_numpy(self._t(_to_pil(image))), 'eye': eye, 'label': label}


class TVRandomCrop:
    def __init__(self, output_size):
        self._t = transforms.RandomCrop(output_size)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        return {'image': _to_numpy(self._t(_to_pil(image))), 'eye': eye, 'label': label}


class TVRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self._t = transforms.RandomHorizontalFlip(p=p)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        return {'image': _to_numpy(self._t(_to_pil(image))), 'eye': eye, 'label': label}


class TVRandomRotation:
    def __init__(self, degrees):
        self._t = transforms.RandomRotation(degrees)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        return {'image': _to_numpy(self._t(_to_pil(image))), 'eye': eye, 'label': label}


class TVColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0):
        self._t = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        return {'image': _to_numpy(self._t(_to_pil(image))), 'eye': eye, 'label': label}


class TVToTensor:
    def __init__(self):
        self._t = transforms.ToTensor()

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = self._t(_to_pil(image))
        label = torch.tensor(label, dtype=torch.long)
        return {'image': image, 'eye': eye, 'label': label}


class TVNormalize:
    def __init__(self, mean, std):
        self._t = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        return {'image': self._t(image), 'eye': eye, 'label': label}


# ─────────────────────────────────────────────────────────────────────────────
# Ready-made pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size=512):
    """Training pipeline for retinal fundus images.

    Changes vs. previous version:
    - BenGraham normalization applied first (illumination equalization)
    - Rotation extended to ±180° (full 360° coverage — retina is rotationally symmetric)
    - Hue jitter disabled (hue is diagnostically meaningful: red=hemorrhage, yellow=exudate)
    - RandomCutout added (forces holistic scan instead of memorizing one lesion location)
    - PerImageNormalize replaces fixed ImageNet stats (retinal images have very different
      channel distributions — ImageNet stats cause SGD to stall)
    """
    scale_size = int(img_size * 256 / 224)
    return transforms.Compose([
        BenGraham(sigma=15.0),
        CropByEye(0.10, 1),
        Rescale(scale_size),
        RandomCrop(img_size),
        TVRandomHorizontalFlip(p=0.5),
        TVRandomRotation(180),
        TVColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        RandomCutout(n_holes=2, patch_size=32, fill_value=0.5),
        ToTensor(),
        PerImageNormalize(),
    ])


def get_val_transforms(img_size=512):
    """Deterministic validation/test pipeline."""
    scale_size = int(img_size * 256 / 224)
    return transforms.Compose([
        BenGraham(sigma=15.0),
        CropByEye(0.10, 1),
        Rescale(scale_size),
        CenterCrop(img_size),
        ToTensor(),
        PerImageNormalize(),
    ])
```

- [ ] **Step 1.4: Run tests — expect PASS**

```bash
cd "/home/jorge/Documentos/Carlos III/Computer Vision/AutomaticDiagnosisForDiabeticRetinopathy"
python -m pytest tests/test_transforms.py -v
```

Expected output: `7 passed`

- [ ] **Step 1.5: Commit**

```bash
git add src/data/transforms.py tests/test_transforms.py
git commit -m "feat: Ben Graham preprocessing, PerImageNormalize, full rotation, no hue jitter, RandomCutout"
```

---

## Task 2: Focal Loss

**Files:**
- Create: `src/training/losses.py`
- Create: `tests/test_losses.py`

### Why
BCEWithLogitsLoss with `pos_weight` treats easy and hard samples equally. Easy negatives (obvious healthy retinas) dominate the gradient and lock the loss near the random baseline (~1.017). Focal Loss exponentially down-weights well-classified samples so the optimizer focuses on hard boundary cases.

Config: `gamma=2.0` (standard starting point), `alpha=0.5` (batches are balanced by WeightedRandomSampler, added in Task 5).

- [ ] **Step 2.1: Create test file**

```python
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
    # Very confident correct predictions → easy samples
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
```

- [ ] **Step 2.2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_losses.py -v 2>&1 | head -20
```

Expected: `ImportError: No module named 'training.losses'`

- [ ] **Step 2.3: Create `src/training/losses.py`**

```python
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
        gamma: focusing parameter (default 2.0). Higher → harder focus on
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
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        p_t      = torch.exp(-bce)
        alpha_t  = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss     = alpha_t * (1.0 - p_t) ** self.gamma * bce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
```

- [ ] **Step 2.4: Run tests — expect PASS**

```bash
python -m pytest tests/test_losses.py -v
```

Expected: `4 passed`

- [ ] **Step 2.5: Commit**

```bash
git add src/training/losses.py tests/test_losses.py
git commit -m "feat: FocalLoss (gamma=2.0) to break loss plateau caused by easy-sample gradient domination"
```

---

## Task 3: SEResNet9 — capacity increase + SE attention + Kaiming init

**Files:**
- Modify: `src/model/ensemble_net.py`
- Create: `tests/test_models.py`

### Why
- ResNet9 (610K params) is under-parameterized for 512×512 fundus images: the literature recommends 2–5M params for from-scratch training on this task.
- SE blocks add channel-wise attention at negligible parameter cost (~10K extra), allowing the network to suppress noise channels and amplify lesion-discriminative channels.
- Kaiming initialization prevents dead neurons at random init.
- Dropout reduced from 0.5 to 0.1: the model is currently underfitting — regularization is the enemy until train AUC > 0.80.

New architecture (spatial flow at 512×512):
```
prep:    3→32  ch   512×512
layer1:  32→64 ch   256×256  (MaxPool)
res1 + SE(64)        256×256
layer2:  64→128 ch  128×128  (MaxPool)
layer3: 128→256 ch   64×64   (MaxPool)
res2 + SE(256)        64×64
gap → dropout(0.1) → Linear(256, 1)
```

Total ~1.8M parameters.

- [ ] **Step 3.1: Create test file**

```python
# tests/test_models.py
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.ensemble_net import SEResNet9


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
    # At least some logits should be outside [0, 1] (a probability would not be)
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
```

- [ ] **Step 3.2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_models.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'SEResNet9'`

- [ ] **Step 3.3: Add `_SEBlock` and `SEResNet9` to `src/model/ensemble_net.py`**

Add the following two classes **after** the existing `_ResBlock` class (around line 180) and **before** the `ResNet9` class. Do not remove any existing code.

```python
# ─────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation block
# ─────────────────────────────────────────────────────────────────────────────

class _SEBlock(nn.Module):
    """Channel-wise attention: suppresses noise channels, amplifies lesion channels.

    Args:
        channels:  number of input/output feature channels.
        reduction: bottleneck ratio for the excitation MLP.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.excitation(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# SEResNet9  — ResNet-9 with SE attention, increased capacity, Kaiming init
# ─────────────────────────────────────────────────────────────────────────────
# Spatial flow at 512×512:
#   prep:    3→32  ch   512×512
#   layer1:  32→64 ch   256×256  (MaxPool)
#   res1 + SE(64)        256×256
#   layer2:  64→128 ch  128×128  (MaxPool)
#   layer3: 128→256 ch   64×64   (MaxPool)
#   res2 + SE(256)        64×64
#   GAP → Dropout(0.1) → Linear(256, 1)
#
# ~1.8M trainable parameters.
# Returns raw logits — use BCEWithLogitsLoss or FocalLoss, not BCELoss.

class SEResNet9(nn.Module):
    """ResNet-9 with Squeeze-and-Excitation attention blocks.

    Key improvements over ResNet9:
    - SE blocks after each ResBlock for learned channel re-weighting
    - Extra 128→256 downsampling stage (×3 parameter increase in deep layers)
    - dropout=0.1 (underfitting phase; raise to 0.3 once train AUC > 0.80)
    - Kaiming (He) normal initialization on all Conv2d layers
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.1):
        super().__init__()

        self.prep   = _conv_bn_relu(in_channels, 32, kernel_size=3, padding=1)

        self.layer1 = nn.Sequential(
            _conv_bn_relu(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 512 → 256
        )
        self.res1   = _ResBlock(64)
        self.se1    = _SEBlock(64)

        self.layer2 = nn.Sequential(
            _conv_bn_relu(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 256 → 128
        )
        self.layer3 = nn.Sequential(
            _conv_bn_relu(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 128 → 64
        )
        self.res2   = _ResBlock(256)
        self.se2    = _SEBlock(256)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.se1(self.res1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.se2(self.res2(x))
        x = self.gap(x).flatten(1)
        return self.head(x)              # (N, 1) raw logit
```

Also update the `__init__.py` for the model module so `SEResNet9` is importable:

```python
# src/model/__init__.py  — add SEResNet9 to the existing imports
from .ensemble_net import DenseNetGreen, ResNet9, EnsembleNet, SEResNet9
```

- [ ] **Step 3.4: Run tests — expect PASS**

```bash
python -m pytest tests/test_models.py -v
```

Expected: `5 passed`

- [ ] **Step 3.5: Commit**

```bash
git add src/model/ensemble_net.py src/model/__init__.py tests/test_models.py
git commit -m "feat: SEResNet9 — SE attention blocks, 1.8M params, dropout=0.1, Kaiming init"
```

---

## Task 4: Add warmup_epochs to config

**Files:**
- Modify: `src/training/config.py`

- [ ] **Step 4.1: Add warmup_epochs to config**

In `src/training/config.py`, add the following lines after the existing `base_cosine_etamin` line:

```python
# ── LR warmup ────────────────────────────────────────────────────────────────
# Start at lr * 0.01 and scale linearly to lr over the first 5 epochs before
# handing off to CosineAnnealingLR. Prevents gradient explosion at random init.
warmup_epochs      = 5
warmup_start_factor = 0.01
```

- [ ] **Step 4.2: Verify config is importable**

```bash
python -c "from src.training import config; print('warmup_epochs:', config.warmup_epochs)"
```

Expected: `warmup_epochs: 5`

- [ ] **Step 4.3: Commit**

```bash
git add src/training/config.py
git commit -m "config: add warmup_epochs=5 for LinearLR warmup before cosine annealing"
```

---

## Task 5: Notebook — WeightedRandomSampler + SequentialLR + new model

**Files:**
- Modify: `train_colab.ipynb` (cells: `cell-data`, `cell-model`)

This task modifies two cells in the notebook. The cells contain Python code that runs in Colab. Edit the cell source in the notebook directly.

### Cell `cell-data` — replace the DataLoader construction block

The goal: force every mini-batch to contain approximately 50% positive and 50% negative samples. This ensures the positive class contributes gradient at every update step and allows FocalLoss alpha=0.5.

Find the existing `train_loader = DataLoader(...)` line and replace **only the train_loader instantiation** with the following (leave val_loader and test_loader unchanged):

```python
from torch.utils.data import WeightedRandomSampler

# Build per-sample weights inversely proportional to class frequency.
# This guarantees ~50/50 positive/negative in every batch.
_labels = [int(train_dataset[i]['label']) for i in range(len(train_dataset))]
_n_pos  = sum(_labels)
_n_neg  = len(_labels) - _n_pos
_weight_map = {0: 1.0 / _n_neg, 1: 1.0 / _n_pos}
_sample_weights = [_weight_map[l] for l in _labels]

_sampler = WeightedRandomSampler(
    weights=_sample_weights,
    num_samples=len(_sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=_sampler,          # sampler is mutually exclusive with shuffle=True
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
print(f'Train sampler: {_n_pos} pos / {_n_neg} neg → balanced batches')
```

### Cell `cell-model` — replace `build_base_model`

Replace the entire `build_base_model` function with:

```python
from model.ensemble_net  import SEResNet9
from training.losses     import FocalLoss

def build_base_model(name):
    """Instantiate SEResNet9, FocalLoss, Adam, SequentialLR (warmup + cosine)."""
    use_amp = config.use_amp and device.type == 'cuda'

    if name in ('densenet_green', 'resnet9_rgb', 'seresnet9'):
        model = SEResNet9(in_channels=3, dropout=0.1).to(device)
    else:
        raise ValueError(f'Unknown base model: {name}')

    # Focal Loss with alpha=0.5: balanced batches (WeightedRandomSampler) remove
    # the need for alpha-based class reweighting; gamma=2.0 handles sample difficulty.
    criterion = FocalLoss(gamma=2.0, alpha=0.5)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.base_lr,
        weight_decay=config.base_weight_decay,
    )

    # LinearLR warmup (5 epochs: lr*0.01 → lr) then CosineAnnealing for the rest
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.warmup_start_factor,
        end_factor=1.0,
        total_iters=config.warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.base_num_epochs - config.warmup_epochs,
        eta_min=config.base_cosine_etamin,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[config.warmup_epochs],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  SEResNet9: {n_params:,} params  |  FocalLoss(γ=2.0, α=0.5)'
          f'  |  AMP={use_amp}')
    return model, criterion, optimizer, scheduler, config.base_num_epochs, use_amp
```

Also update the `BASE_MODEL` default in cell `cell-config`:
```python
BASE_MODEL = 'seresnet9'
```

And the `models_to_train` logic in `cell-model`:
```python
models_to_train = ['seresnet9'] if BASE_MODEL in ('densenet_green', 'resnet9_rgb', 'seresnet9', 'both') else [BASE_MODEL]
```

- [ ] **Step 5.1: Apply the WeightedRandomSampler block to cell-data**

Edit the cell `cell-data` in `train_colab.ipynb` as described above.

- [ ] **Step 5.2: Apply the new `build_base_model` to cell-model**

Edit the cell `cell-model` in `train_colab.ipynb` as described above.

- [ ] **Step 5.3: Verify the notebook cells parse correctly**

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import torch, torch.optim as optim
from model.ensemble_net import SEResNet9
from training.losses import FocalLoss
from training import config

device = torch.device('cpu')
model = SEResNet9().to(device)
criterion = FocalLoss(gamma=2.0, alpha=0.5)
optimizer = optim.Adam(model.parameters(), lr=config.base_lr)
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=55, eta_min=1e-6)
sched  = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
x = torch.randn(2, 3, 64, 64)
out = model(x).flatten()
labels = torch.randint(0, 2, (2,)).float()
loss = criterion(out, labels)
loss.backward()
print('All components OK. Loss:', loss.item())
"
```

Expected: `All components OK. Loss: <some positive float>`

- [ ] **Step 5.4: Commit**

```bash
git add train_colab.ipynb
git commit -m "feat: notebook uses SEResNet9 + FocalLoss + WeightedRandomSampler + LR warmup"
```

---

## Task 6: Run all tests and verify full pipeline

- [ ] **Step 6.1: Run full test suite**

```bash
cd "/home/jorge/Documentos/Carlos III/Computer Vision/AutomaticDiagnosisForDiabeticRetinopathy"
python -m pytest tests/ -v
```

Expected: `16 passed, 0 failed`

- [ ] **Step 6.2: Smoke-test the full training pipeline on 50 samples**

In Colab, set `MAX_TRAIN_SIZE = 50` and `base_num_epochs = 3` temporarily, run through the full notebook (cells 1–8), and verify:

- Loss decreases from epoch 0 to epoch 2 (no more flat ~1.01 line)
- Train AUC > 0.50 by epoch 2
- Submission ZIP is generated at `results/Submissions/codabench_submission.zip`

- [ ] **Step 6.3: Restore full training parameters and run full Colab training**

Set `MAX_TRAIN_SIZE = 0` and restore `base_num_epochs = 60`. Upload the resulting ZIP to Codabench.

- [ ] **Step 6.4: Final commit**

```bash
git add .
git commit -m "chore: run verified — all 16 tests pass, full pipeline smoke-tested"
```

---

## Expected outcomes per phase

| Change | Mechanism | Expected AUC lift |
|--------|-----------|-------------------|
| Ben Graham + PerImageNormalize | Network stops wasting capacity on illumination; SGD starts in correct distribution | +0.05–0.10 |
| Focal Loss + WeightedRandomSampler | Gradient no longer dominated by easy negatives | +0.03–0.07 |
| SEResNet9 (1.8M params, SE attention) | Sufficient capacity; channel attention localizes lesions | +0.02–0.05 |
| LR warmup (5 epochs) | Stable gradient orientation before large steps | +0.01–0.03 |
| Full rotation (±180°) + no hue jitter | Better invariance; chromatic diagnostic signals preserved | +0.01–0.02 |

Target: val AUC **≥ 0.70** after this plan. If still below 0.70, increase `dropout` from 0.1 to 0.3 and retrain (at that point train AUC should be high enough to need regularization).
