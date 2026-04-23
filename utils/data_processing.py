# =============================================================================
# data_processing.py  —  Dataset, transforms and DataLoader factory
# =============================================================================
"""
Pipeline overview
─────────────────
RetinopathyDataset  →  __getitem__ returns a dict {'image', 'eye', 'label'}
    └─ right-eye images are horizontally mirrored on load
    └─ labels are binarised: (label > 0) → 1
    └─ accepts either a csv_file path OR a DataFrame (for K-Fold support)

Transform pipeline (training):
    1. CropByEye              – crop tight around the fundus disc
    2. Rescale                – shortest side to rescale_size (e.g. 580)
    3. DualChannelEnhancement – CLAHE on green + red, border mask preserved
    4. RandomCrop(512)        – random spatial crop to final network size
    5. RandomHFlip / VFlip    – rigid augmentation (aug_rigid=True)
    6. RandomRotation         – up to ±30° (scaled by intensity)
    7. ColorJitter            – brightness, contrast, saturation, hue
    8. GaussianNoise          – sensor noise simulation (aug_regularize=True)
    9. ToTensor + Normalize   – C×H×W float32, ImageNet stats

Transform pipeline (eval / test):
    1. CropByEye
    2. Rescale
    3. DualChannelEnhancement (no random flips / noise)
    4. CenterCrop(512)
    5. ToTensor + Normalize

get_dataloader() builds a WeightedRandomSampler to balance DR / No-DR.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from skimage import io, transform as sk_transform, util as sk_util, color as sk_color

from utils.config import Config as cfg


# =============================================================================
# Low-level CLAHE helpers
# =============================================================================

def apply_clahe_green(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the green channel of an RGB image.

    DR lesions (micro-aneurysms, haemorrhages, exudates) have the highest
    contrast in the green channel under fundus illumination.

    Args:
        image: H×W×3 array, dtype float64 in [0,1] OR uint8 in [0,255].
    Returns:
        Same shape/dtype array with the green channel equalised.
    """
    is_float = image.dtype != np.uint8
    img_u8 = sk_util.img_as_ubyte(np.clip(image, 0, 1)) if is_float else image.copy()

    clahe       = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_grid)
    img_out     = img_u8.copy()
    img_out[:, :, 1] = clahe.apply(img_u8[:, :, 1])   # channel 1 = green

    return sk_util.img_as_float(img_out) if is_float else img_out


def apply_clahe_red(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the red channel of an RGB image.

    The red channel captures haemorrhages and neovascularisation well.

    Args:
        image: H×W×3 array, dtype float64 in [0,1] OR uint8 in [0,255].
    Returns:
        Same shape/dtype array with the red channel equalised.
    """
    is_float = image.dtype != np.uint8
    img_u8 = sk_util.img_as_ubyte(np.clip(image, 0, 1)) if is_float else image.copy()

    clahe       = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_grid)
    img_out     = img_u8.copy()
    img_out[:, :, 0] = clahe.apply(img_u8[:, :, 0])   # channel 0 = red

    return sk_util.img_as_float(img_out) if is_float else img_out


# =============================================================================
# Custom transform classes  (all operate on sample dicts)
# =============================================================================

class CropByEye:
    """Crop tightly around the fundus using a brightness threshold.

    Removes the large black border common in fundus photography so the network
    focuses on retinal tissue rather than background.
    """

    def __init__(self, threshold: float = 0.10, border: int = 5):
        self.threshold = threshold
        self.border    = (border, border) if isinstance(border, int) else border

    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample["image"], sample["eye"], sample["label"]
        h, w = image.shape[:2]
        imgray = sk_color.rgb2gray(image)
        _, mask = cv2.threshold(imgray, self.threshold, 1, cv2.THRESH_BINARY)
        sidx = np.nonzero(mask)
        if len(sidx[0]) < 20:              # malformed / nearly black image
            return sample
        minx  = max(sidx[1].min() - self.border[1], 0)
        maxx  = min(sidx[1].max() + 1 + self.border[1], w)
        miny  = max(sidx[0].min() - self.border[0], 0)
        maxy  = min(sidx[0].max() + 1 + self.border[0], h)
        image = image[miny:maxy, minx:maxx, ...]
        return {"image": image, "eye": eye, "label": label}


class Rescale:
    """Resize so the shortest side equals output_size (int) or exact (h, w)."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample["image"], sample["eye"], sample["label"]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            s = self.output_size
            new_h, new_w = (int(s * h / w), s) if h > w else (s, int(s * w / h))
        else:
            new_h, new_w = self.output_size
        image = sk_transform.resize(image, (new_h, new_w), anti_aliasing=True)
        return {"image": image, "eye": eye, "label": label}


class DualChannelEnhancement:
    """Apply masked CLAHE to both green and red channels simultaneously.

    Rationale
    ─────────
    • Green channel: highest contrast for micro-aneurysms, exudates, haemorrhages.
    • Red channel  : highlights neovascularisation and larger haemorrhages.
    • Strict masking: the black fundus border is preserved by zeroing pixels
      outside a brightness-derived binary mask AFTER enhancement.  This prevents
      CLAHE from amplifying noise in the dark border region.

    Design
    ──────
    1. Compute binary mask: pixels with luminance > mask_threshold are retina.
    2. Apply CLAHE to green channel.
    3. Apply CLAHE to red channel.
    4. Re-apply mask: border pixels are forced back to 0 on all channels.
    """

    def __init__(self, mask_threshold: float = 0.05):
        # A slightly lower threshold than CropByEye to catch the full disc edge
        self.mask_threshold = mask_threshold

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]   # float64 H×W×3 in [0, 1]

        # ── Step 1: Build fundus mask (retina = 1, black border = 0) ─────────
        gray = sk_color.rgb2gray(image)
        mask = (gray > self.mask_threshold).astype(np.float64)  # H×W binary

        # ── Step 2: Enhance green channel ────────────────────────────────────
        image = apply_clahe_green(image)

        # ── Step 3: Enhance red channel ───────────────────────────────────────
        image = apply_clahe_red(image)

        # ── Step 4: Re-apply mask to preserve original black border ──────────
        # Expand mask to (H, W, 1) and broadcast over 3 channels
        image = image * mask[:, :, np.newaxis]

        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


# ── Legacy single-channel wrappers (kept for EDA / backward compatibility) ───

class CLAHEGreenChannel:
    """Apply CLAHE to the green channel only (no masking)."""
    def __call__(self, sample: dict) -> dict:
        image = apply_clahe_green(sample["image"])
        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


class CLAHERedChannel:
    """Apply CLAHE to the red channel only (no masking)."""
    def __call__(self, sample: dict) -> dict:
        image = apply_clahe_red(sample["image"])
        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


# ── Spatial / geometric transforms ───────────────────────────────────────────

class RandomHFlip:
    """Random horizontal flip with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if np.random.rand() < self.p:
            image = sample["image"][:, ::-1, :].copy()
            return {"image": image, "eye": sample["eye"], "label": sample["label"]}
        return sample


class RandomVFlip:
    """Random vertical flip with probability p."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        if np.random.rand() < self.p:
            image = sample["image"][::-1, :, :].copy()
            return {"image": image, "eye": sample["eye"], "label": sample["label"]}
        return sample


class RandomRotation:
    """Random rotation in [-degrees, +degrees]."""
    def __init__(self, degrees: float = 15):
        self.degrees = degrees

    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample["image"], sample["eye"], sample["label"]
        angle = np.random.uniform(-self.degrees, self.degrees)
        pil   = Image.fromarray(sk_util.img_as_ubyte(np.clip(image, 0, 1)))
        pil   = transforms.functional.rotate(pil, angle)
        image = sk_util.img_as_float(np.asarray(pil))
        return {"image": image, "eye": eye, "label": label}


class RandomCrop:
    """Randomly crop to output_size (int → square, or (h, w) tuple).

    NOTE: must be applied AFTER Rescale so the image is always larger
    than the crop target (no padding needed).
    """
    def __init__(self, output_size):
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else output_size
        )

    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample["image"], sample["eye"], sample["label"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top  = np.random.randint(0, max(h - new_h, 0) + 1)
        left = np.random.randint(0, max(w - new_w, 0) + 1)
        image = image[top: top + new_h, left: left + new_w]
        return {"image": image, "eye": eye, "label": label}


class CenterCrop:
    """Center-crop to output_size (int → square, or (h, w) tuple).

    NOTE: must be applied AFTER Rescale so the image is always larger
    than the crop target (no padding needed).
    """
    def __init__(self, output_size):
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else output_size
        )

    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample["image"], sample["eye"], sample["label"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top  = max((h - new_h) // 2, 0)
        left = max((w - new_w) // 2, 0)
        image = image[top: top + new_h, left: left + new_w]
        return {"image": image, "eye": eye, "label": label}


# ── Pixel-level transforms ────────────────────────────────────────────────────

class GaussianNoise:
    """Add zero-mean Gaussian noise; simulates sensor variability."""
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        noise = np.random.normal(0, self.std, image.shape).astype(image.dtype)
        image = np.clip(image + noise, 0, 1)
        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


class ColorJitter:
    """Random brightness / contrast / saturation / hue jitter (torchvision)."""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05):
        self.cj = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample["image"], sample["eye"], sample["label"]
        pil   = Image.fromarray(sk_util.img_as_ubyte(np.clip(image, 0, 1)))
        image = sk_util.img_as_float(np.asarray(self.cj(pil)))
        return {"image": image, "eye": eye, "label": label}


class ToTensor:
    """numpy H×W×C float64 → torch C×H×W float32; label → long tensor."""
    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample["image"], sample["eye"], sample["label"]
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        label = torch.tensor(label, dtype=torch.long)
        return {"image": image, "eye": eye, "label": label}


class Normalize:
    """Per-channel ImageNet normalisation on a C×H×W float tensor."""
    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(mean or cfg.pixel_mean, dtype=torch.float32)
        self.std  = torch.tensor(std  or cfg.pixel_std,  dtype=torch.float32)

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        image = image.sub(self.mean[:, None, None]).div(self.std[:, None, None])
        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


# =============================================================================
# Transform factories
# =============================================================================

def build_train_transforms(
    img_size:     int   = cfg.img_height,
    rescale_size: int   = cfg.rescale_size,
    rigid:        bool  = cfg.aug_rigid,
    regularize:   bool  = cfg.aug_regularize,
    clahe:        bool  = cfg.aug_clahe,
    intensity:    float = cfg.aug_intensity,
) -> transforms.Compose:
    """Build the augmented training pipeline.

    All magnitude-based transforms are scaled by ``intensity`` ∈ [0.0, 1.0].

    Pipeline order:
        CropByEye → Rescale → DualChannelEnhancement →
        RandomCrop(img_size) → [rigid aug] → ColorJitter → [noise]
        → ToTensor → Normalize

    The fixed-size guarantee: Rescale raises the shorter side to rescale_size
    (> img_size), so RandomCrop always has pixels to spare — no padding.
    """
    intensity = float(np.clip(intensity, 0.0, 1.0))

    steps = [
        CropByEye(threshold=0.10, border=5),
        Rescale(rescale_size),              # guarantees image ≥ img_size on both sides
    ]

    # Dual CLAHE with border masking (applied BEFORE crop to avoid edge artefacts)
    if clahe:
        steps.append(DualChannelEnhancement(mask_threshold=0.05))

    # RandomCrop to final training resolution
    steps.append(RandomCrop(img_size))

    # Rigid spatial augmentation
    if rigid:
        steps += [
            RandomHFlip(p=0.5),
            RandomVFlip(p=0.3 * intensity),
            RandomRotation(degrees=30.0 * intensity),
        ]

    # Color perturbation
    steps.append(ColorJitter(
        brightness=0.50 * intensity,
        contrast  =0.50 * intensity,
        saturation=0.30 * intensity,
        hue       =0.10 * intensity,
    ))

    # Additive Gaussian noise
    if regularize:
        steps.append(GaussianNoise(std=0.03 * intensity))

    steps += [ToTensor(), Normalize()]
    return transforms.Compose(steps)


def build_eval_transforms(
    img_size:     int  = cfg.img_height,
    rescale_size: int  = cfg.rescale_size,
    clahe:        bool = cfg.aug_clahe,
) -> transforms.Compose:
    """Deterministic evaluation / test pipeline (no random augmentation).

    CropByEye → Rescale → DualChannelEnhancement → CenterCrop → ToTensor → Normalize
    """
    steps = [
        CropByEye(threshold=0.10, border=5),
        Rescale(rescale_size),
    ]
    if clahe:
        steps.append(DualChannelEnhancement(mask_threshold=0.05))
    steps += [CenterCrop(img_size), ToTensor(), Normalize()]
    return transforms.Compose(steps)


def build_tta_transforms(
    img_size:     int  = cfg.img_height,
    rescale_size: int  = cfg.rescale_size,
) -> transforms.Compose:
    """Light stochastic pipeline for Test-Time Augmentation (TTA).

    Uses mild flips and rotation — no aggressive colour jitter or noise —
    to avoid distorting test predictions while still adding useful diversity.
    """
    steps = [
        CropByEye(threshold=0.10, border=5),
        Rescale(rescale_size),
        DualChannelEnhancement(mask_threshold=0.05),
        RandomCrop(img_size),
        RandomHFlip(p=0.5),
        RandomVFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        ToTensor(),
        Normalize(),
    ]
    return transforms.Compose(steps)


# =============================================================================
# Dataset
# =============================================================================

class RetinopathyDataset(Dataset):
    """PyTorch Dataset for the Diabetic Retinopathy fundus image challenge.

    Accepts either a CSV file path OR a pandas DataFrame (for K-Fold splits
    where the fold indices are pre-computed in the notebook).

    Args:
        csv_file:  Path to the split CSV.  Ignored when ``dataframe`` is given.
        root_dir:  Root folder containing the ``images/`` sub-directory.
        transform: Callable that receives and returns a sample dict.
        max_size:  If > 0, randomly sub-sample to this many images (dev mode).
        dataframe: Optional DataFrame to use instead of reading csv_file.
                   Must contain columns: id (str), eye (int), label (int).
    """

    def __init__(
        self,
        csv_file:  str | None   = None,
        root_dir:  str          = str(cfg.data_dir),
        transform              = None,
        max_size:  int          = 0,
        dataframe: pd.DataFrame | None = None,
    ):
        # Accept either a pre-built DataFrame (K-Fold) or a CSV path
        if dataframe is not None:
            self.df = dataframe.reset_index(drop=True)
        elif csv_file is not None:
            self.df = pd.read_csv(
                csv_file,
                header=0,
                dtype={"id": str, "eye": int, "label": int},
            )
        else:
            raise ValueError("Provide either csv_file or dataframe.")

        if max_size > 0:
            rng = np.random.RandomState(seed=cfg.seed)
            idx = rng.permutation(len(self.df))[:max_size]
            self.df = self.df.iloc[idx].reset_index(drop=True)

        self.img_dir   = os.path.join(root_dir, "images")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    @property
    def binary_labels(self) -> np.ndarray:
        """Binary labels (0 / 1) for every sample — used for WeightedRandomSampler."""
        raw = self.df["label"].values
        return (raw > 0).astype(np.int64)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.item()

        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row["id"]) + ".jpg")
        image    = io.imread(img_path)                   # H×W×3 uint8

        # Right-eye images are mirrored for anatomical consistency
        if row["eye"] == 1:
            image = image[:, ::-1, :].copy()

        # Convert to float64 in [0, 1] for the custom transform pipeline
        image = sk_util.img_as_float(image)

        # Binarise: label -1 (test) stays -1; 0 stays 0; 1-4 → 1
        raw_label    = int(row["label"])
        binary_label = raw_label if raw_label < 0 else int(raw_label > 0)

        sample = {"image": image, "eye": int(row["eye"]), "label": binary_label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# =============================================================================
# DataLoader factory
# =============================================================================

def get_dataloader(
    csv_file:             str | None     = None,
    root_dir:             str            = str(cfg.data_dir),
    is_train:             bool           = True,
    batch_size:           int            = cfg.batch_size,
    max_size:             int            = 0,
    transform                            = None,
    use_weighted_sampler: bool           = True,
    aug_intensity:        float          = cfg.aug_intensity,
    dataframe:            pd.DataFrame | None = None,
) -> tuple[RetinopathyDataset, DataLoader]:
    """Build a Dataset + DataLoader for one split.

    For training splits the function:
        1. Counts class frequencies.
        2. Assigns each sample a weight = 1 / class_count.
        3. Uses WeightedRandomSampler so DR and No-DR appear with equal probability.

    Args:
        csv_file:             Path to split CSV (or None if dataframe is given).
        root_dir:             Folder containing ``images/``.
        is_train:             True → augmented pipeline + WeightedRandomSampler.
        batch_size:           Batch size.
        max_size:             Sub-sample size (0 = all images).
        transform:            Override the auto-selected transform pipeline.
        use_weighted_sampler: False → plain random shuffling.
        aug_intensity:        Augmentation magnitude for build_train_transforms.
        dataframe:            Optional DataFrame for K-Fold splits.

    Returns:
        (dataset, dataloader) tuple.
    """
    if transform is None:
        transform = (
            build_train_transforms(intensity=aug_intensity)
            if is_train else
            build_eval_transforms()
        )

    dataset = RetinopathyDataset(
        csv_file=csv_file, root_dir=root_dir,
        transform=transform, max_size=max_size,
        dataframe=dataframe,
    )

    sampler = None
    shuffle = False

    if is_train and use_weighted_sampler:
        bin_labels    = dataset.binary_labels               # shape (N,)
        class_counts  = np.bincount(bin_labels)             # [count_0, count_1]
        class_weights = 1.0 / class_counts.astype(float)   # inverse frequency
        sample_weights = class_weights[bin_labels]          # per-sample weight
        sampler = WeightedRandomSampler(
            weights     = torch.from_numpy(sample_weights).double(),
            num_samples = len(dataset),
            replacement = True,
        )
        print(
            f"[DataLoader] {os.path.basename(str(csv_file or 'DataFrame'))} | "
            f"No-DR: {class_counts[0]:,} | DR: {class_counts[1]:,} | "
            f"WeightedSampler ratio ≈ 1:{class_counts[0] / class_counts[1]:.2f}"
        )
    elif is_train:
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        sampler     = sampler,
        shuffle     = shuffle,
        num_workers = 0,
        pin_memory  = cfg.device == "cuda",
        drop_last   = is_train,   # prevents 1-sample batches that break BatchNorm
    )
    return dataset, loader