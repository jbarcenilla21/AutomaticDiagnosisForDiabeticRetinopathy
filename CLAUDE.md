# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University lab session (UC3M Computer Vision Master) — binary classification of Diabetic Retinopathy (DR) from retinal fundus images using PyTorch CNNs. The task converts the original 5-class severity scale (0–4) into a binary label: **0 = No DR, 1 = DR**.

Two submission tracks for Codabench:
- **Custom**: CNN built from scratch, no pretrained weights
- **Fine-tuning**: Pretrained torchvision model fine-tuned on DR data

## Setup

```bash
pip install -r requirements.txt
```

Dataset goes under `data/` (not tracked by git): `data/images/`, `data/train.csv`, `data/val.csv`, `data/test.csv`.

## Running the Notebook

```bash
jupyter notebook LS5_CV_CNNs_Retinopathy.ipynb
```

The notebook is the single deliverable — both Custom and Fine-tuning models must live inside it. The supporting `.py` files are helpers imported by the notebook.

## Architecture

### Data Pipeline (defined in the notebook)

`RetinopathyDataset` (subclass of `torch.utils.data.Dataset`) handles:
- Reading CSV with columns `id` (5-digit string), `eye` (0=left, 1=right), `label` (0–4, or -1 for test)
- Right-eye images are horizontally mirrored (`image[:,::-1,:]`) before any transform
- Labels are binarized: `(label > 0).astype(int)`
- Optional `maxSize` parameter subsamples the dataset for fast iteration during development

Custom transform classes (callable objects, also defined in the notebook): `CropByEye`, `Rescale`, `RandomCrop`, `CenterCrop`, `ToTensor`, `Normalize`. Samples flow as dicts `{'image': ..., 'eye': ..., 'label': ...}`.

### Supporting Modules

| File | Purpose |
|------|---------|
| `utils/config.py` | Hyperparameters: `learning_rate=1e-3`, `batch_size=32`, `num_epochs=50`, `img_height/width=224` |
| `utils/utils.py` | `get_transforms(train)` — torchvision-based transforms with ImageNet normalization (`mean=[0.485,0.456,0.406]`) |
| `model/model.py` | `Trainer` class with `train_epoch(dataloader)` and `evaluate(dataloader)` methods |
| `model/model_components.py` | Placeholder for reusable CNN building blocks |

### Evaluation

Metric is **AUC** (area under the ROC curve), computed with `sklearn.metrics`. The test set has no labels (`-1`); the submission is a 1000×1 CSV of predicted DR scores (not hard labels).

### Reproducibility

Seeds are fixed in the notebook: `random.seed(42)`, `numpy.random.seed(42)`, `torch.manual_seed(42)`, `torch.backends.cudnn.enabled = False`.

## Key Constraints

- The **Custom** track forbids any pretrained weights or external model architectures — the full network code must appear in the notebook.
- Submission outputs go to `outputs/` (not tracked by git).
- Code will be reviewed for reproducibility; non-reproducible results lead to disqualification.
