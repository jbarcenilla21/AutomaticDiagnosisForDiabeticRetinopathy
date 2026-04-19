# =============================================================================
# data_processing.py  —  Dataset, transforms and DataLoader factory
# =============================================================================
"""
Pipeline overview
─────────────────
RetinopathyDataset  →  __getitem__ returns a dict {'image', 'eye', 'label'}
  └─ right-eye images are horizontally mirrored on load
  └─ labels are binarised: (label > 0) → 1

Transform pipeline (toggled via cfg flags):
  1. CropByEye          – crop tight around the fundus disc
  2. Rescale            – shortest side to RESCALE_SIZE
  3. CLAHEGreenChannel  – (AUG_CLAHE=True) CLAHE on the green channel only
  4. Random/Center crop – 224×224
  5. Rigid augmentation – random h-flip, v-flip, rotation  (AUG_RIGID=True)
  6. Regularisation aug – random crop offset + Gaussian noise (AUG_REGULARIZE)
  7. ToTensor + Normalize

get_dataloader()  builds a WeightedRandomSampler to balance DR / No-DR.
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


# -----------------------------------------------------------------------------
# Helper: CLAHE on the green channel
# -----------------------------------------------------------------------------

def apply_clahe_green(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the green channel of an RGB image (uint8 or float64).

    DR lesions (micro-aneurysms, haemorrhages, exudates) have the highest
    contrast in the green channel under fundus illumination.

    Args:
        image: H×W×3 array, dtype float64 in [0,1] OR uint8 in [0,255].

    Returns:
        Same shape/dtype array with the green channel histogram-equalised.
    """
    is_float = image.dtype != np.uint8
    if is_float:
        img_u8 = sk_util.img_as_ubyte(np.clip(image, 0, 1))
    else:
        img_u8 = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip,
        tileGridSize=cfg.clahe_grid,
    )
    # Work in-place on a copy; only modify channel 1 (green)
    img_out = img_u8.copy()
    img_out[:, :, 1] = clahe.apply(img_u8[:, :, 1])

    if is_float:
        return sk_util.img_as_float(img_out)
    return img_out


# -----------------------------------------------------------------------------
# Custom transform classes  (all operate on sample dicts)
# -----------------------------------------------------------------------------

class CropByEye:
    """Crop tightly around the fundus using a brightness threshold.

    Removes the large black border common in fundus photography so the network
    focuses on retinal tissue rather than background.
    """

    def __init__(self, threshold: float = 0.10, border: int = 5):
        self.threshold = threshold
        self.border = (border, border) if isinstance(border, int) else border

    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample['image'], sample['eye'],sample['label']
        h, w = image.shape[:2]
        imgray = sk_color.rgb2gray(image)
        #Compute the mask
        th, mask = cv2.threshold(imgray, self.threshold, 1, cv2.THRESH_BINARY)
        #Compute the coordinates of the bounding box that contains the mask
        sidx=np.nonzero(mask)
        #In case the mask is too small, impies malfunctioning
        if len(sidx[0])<20:
            return {'image': image, 'eye': eye, 'label' : label}
        minx=np.maximum(sidx[1].min()-self.border[1],0)
        maxx=np.minimum(sidx[1].max()+1+self.border[1],w)
        miny=np.maximum(sidx[0].min()-self.border[0],0)
        maxy=np.minimum(sidx[0].max()+1+self.border[1],h)
        #Crop the image
        image=image[miny:maxy,minx:maxx,...]
        return {'image': image, 'eye': eye, 'label' : label}

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


class CLAHEGreenChannel:
    """Apply CLAHE to the green channel to enhance DR lesion visibility."""

    def __call__(self, sample: dict) -> dict:
        image = apply_clahe_green(sample["image"])
        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


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
        pil = Image.fromarray(sk_util.img_as_ubyte(np.clip(image, 0, 1)))
        pil = transforms.functional.rotate(pil, angle)
        image = sk_util.img_as_float(np.asarray(pil))
        return {"image": image, "eye": eye, "label": label}


class RandomCrop:
    """Randomly crop to output_size (int for square, or (h, w) tuple)."""

    def __init__(self, output_size):
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else output_size
        )

    def __call__(self, sample: dict) -> dict:
        image, eye, label = sample["image"], sample["eye"], sample["label"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top  = np.random.randint(0, h - new_h + 1) if h > new_h else 0
        left = np.random.randint(0, w - new_w + 1) if w > new_w else 0
        image = image[top : top + new_h, left : left + new_w]
        return {"image": image, "eye": eye, "label": label}


class CenterCrop:
    """Center-crop to output_size (int for square, or (h, w) tuple)."""

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
        image = image[top : top + new_h, left : left + new_w]
        return {"image": image, "eye": eye, "label": label}


class GaussianNoise:
    """Add zero-mean Gaussian noise; models sensor noise variability."""

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        noise = np.random.normal(0, self.std, image.shape).astype(image.dtype)
        image = np.clip(image + noise, 0, 1)
        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


class ColorJitter:
    """Random brightness / contrast / saturation jitter (via torchvision)."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self.cj = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
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
    """Per-channel normalisation on a CxHxW float tensor."""

    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(mean or cfg.pixel_mean, dtype=torch.float32)
        self.std  = torch.tensor(std  or cfg.pixel_std,  dtype=torch.float32)

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        image = image.sub(self.mean[:, None, None]).div(self.std[:, None, None])
        return {"image": image, "eye": sample["eye"], "label": sample["label"]}


# -----------------------------------------------------------------------------
# Transform factories
# -----------------------------------------------------------------------------

def build_train_transforms(
    img_size: int = cfg.img_height,
    rescale_size: int = cfg.rescale_size,
    rigid: bool = cfg.aug_rigid,
    regularize: bool = cfg.aug_regularize,
    clahe: bool = cfg.aug_clahe,
) -> transforms.Compose:
    """Build the augmented training pipeline."""
    steps = [
        CropByEye(threshold=0.10, border=5),
        Rescale(rescale_size),
    ]
    if clahe:
        steps.append(CLAHEGreenChannel())
    steps.append(RandomCrop(img_size))
    if rigid:
        steps += [
            RandomHFlip(p=0.5),
            RandomVFlip(p=0.3),
            RandomRotation(degrees=20),
        ]
    steps.append(ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05))
    if regularize:
        steps.append(GaussianNoise(std=0.015))
    steps += [ToTensor(), Normalize()]
    return transforms.Compose(steps)


def build_eval_transforms(
    img_size: int = cfg.img_height,
    rescale_size: int = cfg.rescale_size,
    clahe: bool = cfg.aug_clahe,
) -> transforms.Compose:
    """Deterministic evaluation / test pipeline."""
    steps = [
        CropByEye(threshold=0.10, border=5),
        Rescale(rescale_size),
    ]
    if clahe:
        steps.append(CLAHEGreenChannel())
    steps += [CenterCrop(img_size), ToTensor(), Normalize()]
    return transforms.Compose(steps)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class RetinopathyDataset(Dataset):
    """PyTorch Dataset for the Diabetic Retinopathy fundus image challenge.

    Args:
        csv_file:  Path to the split CSV (train / val / test).
        root_dir:  Root folder containing the ``images/`` sub-directory.
        transform: Callable that receives and returns a sample dict.
        max_size:  If > 0, randomly sub-sample to this many images (dev mode).
    """

    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform=None,
        max_size: int = 0,
    ):
        self.df = pd.read_csv(
            csv_file,
            header=0,
            dtype={"id": str, "eye": int, "label": int},
        )
        if max_size > 0:
            rng = np.random.RandomState(seed=cfg.seed)
            idx = rng.permutation(len(self.df))[:max_size]
            self.df = self.df.iloc[idx].reset_index(drop=True)

        self.img_dir  = os.path.join(root_dir, "images")
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

        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["id"] + ".jpg")
        image    = io.imread(img_path)                    # H×W×3 uint8

        # Right-eye images are mirrored for anatomical consistency
        if row["eye"] == 1:
            image = image[:, ::-1, :].copy()

        # Convert to float64 in [0, 1] for the custom transform pipeline
        image = sk_util.img_as_float(image)

        # Binarise: label -1 (test) → kept as -1; 0 stays 0; 1-4 → 1
        raw_label = int(row["label"])
        binary_label = raw_label if raw_label < 0 else int(raw_label > 0)

        sample = {"image": image, "eye": int(row["eye"]), "label": binary_label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# -----------------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------------

def get_dataloader(
    csv_file: str,
    root_dir: str = cfg.data_dir,
    is_train: bool = True,
    batch_size: int = cfg.batch_size,
    max_size: int = 0,
    transform=None,
    use_weighted_sampler: bool = True,
) -> tuple[RetinopathyDataset, DataLoader]:
    """Build a Dataset and DataLoader for one split.

    For training splits the function:
      1. Counts class frequencies.
      2. Assigns each sample a weight = 1 / class_count.
      3. Passes a WeightedRandomSampler to DataLoader so DR and No-DR images
         appear with roughly equal probability per batch.

    Args:
        csv_file:              Path to split CSV.
        root_dir:              Folder containing ``images/``.
        is_train:              True → augmented pipeline + WeightedRandomSampler.
        batch_size:            Batch size.
        max_size:              Sub-sample size (0 = all).
        transform:             Override the auto-selected transform pipeline.
        use_weighted_sampler:  Set False to use plain random shuffling instead.

    Returns:
        (dataset, dataloader) tuple.
    """
    if transform is None:
        transform = build_train_transforms() if is_train else build_eval_transforms()

    dataset = RetinopathyDataset(csv_file, root_dir, transform=transform, max_size=max_size)

    sampler = None
    shuffle = False

    if is_train and use_weighted_sampler:
        bin_labels = dataset.binary_labels          # shape (N,)
        class_counts = np.bincount(bin_labels)      # [count_0, count_1]
        # weight per class: inverse frequency
        class_weights = 1.0 / class_counts.astype(float)
        sample_weights = class_weights[bin_labels]  # shape (N,)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(dataset),
            replacement=True,
        )
        print(
            f"[DataLoader] {os.path.basename(csv_file)} | "
            f"No-DR: {class_counts[0]:,} | DR: {class_counts[1]:,} | "
            f"WeightedSampler ratio ≈ 1:{class_counts[0]/class_counts[1]:.2f}"
        )
    elif is_train:
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory= cfg.device == "cuda",
        drop_last=is_train,        # avoids 1-sample batches causing BN issues
    )
    return dataset, loader