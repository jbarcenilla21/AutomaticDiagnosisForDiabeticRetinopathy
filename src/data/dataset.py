import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import pandas as pd


class RetinopathyDataset(Dataset):
    """Retinal fundus image dataset for binary DR classification.

    Reads a CSV with columns: id (5-digit str), eye (0=left, 1=right), label (0-4 or -1 for test).
    Right-eye images are horizontally mirrored before any transform.
    Labels are binarized: (label > 0) -> 1 (DR), 0 (No DR).
    Test-set labels (-1) are preserved as-is and should not be used for training/eval.
    """

    def __init__(self, csv_file, root_dir, transform=None, maxSize=0):
        self.dataset = pd.read_csv(
            csv_file, header=0, dtype={'id': str, 'eye': int, 'label': int}
        )
        if maxSize > 0:
            idx = np.random.RandomState(seed=42).permutation(range(len(self.dataset)))
            self.dataset = self.dataset.iloc[idx[:maxSize]].reset_index(drop=True)

        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.classes = ['No DR', 'DR']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.dataset.id[idx] + '.jpg')
        image = io.imread(img_path)

        if self.dataset.eye[idx] == 1:
            image = image[:, ::-1, :]

        label = self.dataset.label[idx]
        binary_label = (label > 0).astype(dtype=np.int64) if label >= 0 else label

        sample = {
            'image': image,
            'eye': self.dataset.eye[idx],
            'label': binary_label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
