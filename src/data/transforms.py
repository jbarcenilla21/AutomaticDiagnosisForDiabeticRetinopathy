"""Transform classes for the Retinopathy dataset.

All transforms are callable objects that accept and return a sample dict:
    {'image': np.ndarray (HxWxC), 'eye': int, 'label': int/tensor}

Custom track pipeline (SEResNet9):
  Train: CropByEye → BenGraham → Rescale → RandomCrop → augmentation
         → RandomCutout → ToTensor → PerImageNormalize
  Val:   CropByEye → BenGraham → Rescale → CenterCrop
         → ToTensor → PerImageNormalize

Fine-tune track pipeline (EnsembleModel):
  Train: CropByEye → Rescale → DualChannelEnhancement → RandomCrop
         → flips/rotation → GaussianNoise → ToTensor → Normalize(ImageNet)
  Val:   CropByEye → Rescale → DualChannelEnhancement → CenterCrop
         → ToTensor → Normalize(ImageNet)

NOTE: CropByEye MUST precede BenGraham. BenGraham maps black borders to
gray(128), which breaks CropByEye's brightness-threshold detection.
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
# Fine-tune track: CLAHE helpers + DualChannelEnhancement + GaussianNoise
# ─────────────────────────────────────────────────────────────────────────────

def _apply_clahe_green(image: np.ndarray, clip_limit: float = 2.0,
                       tile_grid: tuple = (4, 4)) -> np.ndarray:
    """Apply CLAHE to the green channel of an RGB image (uint8 or float[0,1])."""
    is_float = image.dtype != np.uint8
    img_u8   = util.img_as_ubyte(np.clip(image, 0, 1)) if is_float else image.copy()
    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    out      = img_u8.copy()
    out[:, :, 1] = clahe.apply(img_u8[:, :, 1])
    return util.img_as_float(out) if is_float else out


def _apply_clahe_red(image: np.ndarray, clip_limit: float = 2.0,
                     tile_grid: tuple = (4, 4)) -> np.ndarray:
    """Apply CLAHE to the red channel of an RGB image (uint8 or float[0,1])."""
    is_float = image.dtype != np.uint8
    img_u8   = util.img_as_ubyte(np.clip(image, 0, 1)) if is_float else image.copy()
    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    out      = img_u8.copy()
    out[:, :, 0] = clahe.apply(img_u8[:, :, 0])
    return util.img_as_float(out) if is_float else out


class DualChannelEnhancement:
    """Apply masked CLAHE to both green and red channels simultaneously.

    Rationale
    ─────────
    • Green channel: highest contrast for micro-aneurysms, exudates, haemorrhages.
    • Red channel  : highlights neovascularisation and larger haemorrhages.
    • Strict masking: the black fundus border is preserved by zeroing pixels
      outside a brightness-derived binary mask AFTER enhancement.

    Used in the Fine-tune track pipeline (EnsembleModel).
    """

    def __init__(self, mask_threshold: float = 0.05,
                 clip_limit: float = 2.0, tile_grid: tuple = (4, 4)):
        self.mask_threshold = mask_threshold
        self.clip_limit     = clip_limit
        self.tile_grid      = tile_grid

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]   # float64 H×W×3 in [0, 1]

        gray = color.rgb2gray(image)
        mask = (gray > self.mask_threshold).astype(np.float64)

        image = _apply_clahe_green(image, self.clip_limit, self.tile_grid)
        image = _apply_clahe_red(image,   self.clip_limit, self.tile_grid)
        image = image * mask[:, :, np.newaxis]

        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


class GaussianNoise:
    """Add zero-mean Gaussian noise to simulate sensor variability.

    Used in the Fine-tune track training pipeline.

    Args:
        std: standard deviation of the noise (default 0.02 for float images).
    """

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        noise = np.random.normal(0, self.std, image.shape).astype(image.dtype)
        image = np.clip(image + noise, 0, 1)
        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


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
        CropByEye(0.10, 1),
        BenGraham(sigma=15.0),
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
    """Deterministic validation/test pipeline (Custom track)."""
    scale_size = int(img_size * 256 / 224)
    return transforms.Compose([
        CropByEye(0.10, 1),
        BenGraham(sigma=15.0),
        Rescale(scale_size),
        CenterCrop(img_size),
        ToTensor(),
        PerImageNormalize(),
    ])


# ImageNet normalization constants for pretrained backbone fine-tuning
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_ft_train_transforms(img_size=512):
    """Training pipeline for the Fine-tune track (EnsembleModel).

    Uses DualChannelEnhancement (CLAHE on green+red) instead of BenGraham,
    ImageNet normalization (correct for pretrained backbones), and GaussianNoise
    for sensor variability augmentation.
    """
    scale_size = int(img_size * 256 / 224)
    return transforms.Compose([
        CropByEye(0.10, 1),
        Rescale(scale_size),
        DualChannelEnhancement(mask_threshold=0.05),
        RandomCrop(img_size),
        TVRandomHorizontalFlip(p=0.5),
        TVRandomRotation(30),
        TVColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        GaussianNoise(std=0.02),
        ToTensor(),
        TVNormalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_ft_val_transforms(img_size=512):
    """Deterministic validation/test pipeline for the Fine-tune track."""
    scale_size = int(img_size * 256 / 224)
    return transforms.Compose([
        CropByEye(0.10, 1),
        Rescale(scale_size),
        DualChannelEnhancement(mask_threshold=0.05),
        CenterCrop(img_size),
        ToTensor(),
        TVNormalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
