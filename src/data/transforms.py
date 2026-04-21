"""Transform classes for the Retinopathy dataset.

All transforms are callable objects that accept and return a sample dict:
    {'image': np.ndarray (HxWxC float64), 'eye': int, 'label': int/tensor}

Two families:
- Custom transforms  : operate directly on numpy arrays (skimage-style).
- TV* wrappers       : bridge numpy <-> PIL to reuse torchvision transforms.

Pipeline order:  CropByEye -> Rescale -> RandomCrop/CenterCrop -> [augmentation] -> ToTensor -> Normalize
"""

import numpy as np
import torch
import cv2
from PIL import Image
from skimage import transform, util, color
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Custom transforms (numpy-based)
# ─────────────────────────────────────────────────────────────────────────────

class CropByEye:
    """Crop the image to the bounding box of the eye, removing black borders.

    Args:
        threshold: pixel intensity threshold in [0, 1] to segment the eye.
        border:    extra pixels to expand the bounding box (int or (h, w) tuple).
    """

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
# torchvision wrappers (numpy -> PIL -> torchvision transform -> numpy/tensor)
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
        image = _to_numpy(self._t(_to_pil(image)))
        return {'image': image, 'eye': eye, 'label': label}


class TVRescale:
    def __init__(self, output_size):
        self._t = transforms.Resize(output_size)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = _to_numpy(self._t(_to_pil(image)))
        return {'image': image, 'eye': eye, 'label': label}


class TVRandomCrop:
    def __init__(self, output_size):
        self._t = transforms.RandomCrop(output_size)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = _to_numpy(self._t(_to_pil(image)))
        return {'image': image, 'eye': eye, 'label': label}


class TVRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self._t = transforms.RandomHorizontalFlip(p=p)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = _to_numpy(self._t(_to_pil(image)))
        return {'image': image, 'eye': eye, 'label': label}


class TVRandomRotation:
    def __init__(self, degrees):
        self._t = transforms.RandomRotation(degrees)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = _to_numpy(self._t(_to_pil(image)))
        return {'image': image, 'eye': eye, 'label': label}


class TVColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05):
        self._t = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        )

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = _to_numpy(self._t(_to_pil(image)))
        return {'image': image, 'eye': eye, 'label': label}


class TVToTensor:
    """numpy float64 (HxWxC) -> torch float32 tensor (CxHxW) via torchvision."""

    def __init__(self):
        self._t = transforms.ToTensor()

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = self._t(_to_pil(image))
        label = torch.tensor(label, dtype=torch.long)
        return {'image': image, 'eye': eye, 'label': label}


class TVNormalize:
    """Normalize a float tensor image. Must follow ToTensor/TVToTensor."""

    def __init__(self, mean, std):
        self._t = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        image, eye, label = sample['image'], sample['eye'], sample['label']
        image = self._t(image)
        return {'image': image, 'eye': eye, 'label': label}


# ─────────────────────────────────────────────────────────────────────────────
# Ready-made pipelines
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size=224):
    """Training pipeline with data augmentation.

    Rescales to 256 (longer margin) before RandomCrop so every crop position
    is valid and the network sees shifted views of the fundus.
    Augmentations chosen for retinal photography:
      - horizontal flip: anatomically symmetric after eye-mirroring
      - rotation ±15°: fundus images have no fixed orientation
      - colour jitter: models inter-device illumination variance
    """
    scale_size = int(img_size * 256 / 224)   # 256 when img_size=224
    return transforms.Compose([
        CropByEye(0.10, 1),
        Rescale(scale_size),
        RandomCrop(img_size),
        TVRandomHorizontalFlip(p=0.5),
        TVRandomRotation(15),
        TVColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size=224):
    """Deterministic validation/test pipeline."""
    scale_size = int(img_size * 256 / 224)
    return transforms.Compose([
        CropByEye(0.10, 1),
        Rescale(scale_size),
        CenterCrop(img_size),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
