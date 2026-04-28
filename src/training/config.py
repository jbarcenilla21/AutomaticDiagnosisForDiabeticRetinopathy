# =============================================================================
# config.py  —  Global configuration for DR classification pipeline
# =============================================================================
"""
Three configuration classes:
  Config         — shared base defaults
  CustomConfig   — from-scratch training (no pretrained weights, SEResNet9)
  FineTuneConfig — transfer learning from ImageNet pretrained models (EnsembleModel)
"""

from pathlib import Path
from torch.cuda import is_available


class Config:

    # ── Hardware & paths ──────────────────────────────────────────────────────
    device = "cuda" if is_available() else "cpu"

    base_dir        = Path(__file__).resolve().parent.parent.parent
    checkpoint_name = Path("best_model.pth")
    checkpoint_dir  = base_dir / "results" / "checkpoints"
    results_dir     = base_dir / "results"
    output_dir      = base_dir / "results" / "outputs"
    train_csv       = base_dir / "data" / "train.csv"
    val_csv         = base_dir / "data" / "val.csv"
    test_csv        = base_dir / "data" / "test.csv"
    data_dir        = base_dir / "data"
    images_dir      = base_dir / "data" / "images"

    # ── Multi-scale ensemble resolution (one scale per member) ───────────────
    ensemble_scales = [224, 384, 512]

    # ── Data parameters ───────────────────────────────────────────────────────
    img_height   = 512
    img_width    = 512
    num_channels = 3
    rescale_size = 580

    # ── Training hyperparameters ──────────────────────────────────────────────
    num_clasess     = 1
    unfreeze_layers = 2
    batch_size      = 16
    num_epochs      = 50
    learning_rate   = 3e-4
    weight_decay    = 1e-4
    lr_patience     = 3
    lr_factor       = 0.3
    early_stopping  = 10
    seed            = 42

    # ── LR warmup (LinearLR → CosineAnnealingLR via SequentialLR) ────────────
    warmup_epochs       = 5
    warmup_start_factor = 0.01

    # ── TTA ───────────────────────────────────────────────────────────────────
    num_tta = 10

    # ── Focal Loss ────────────────────────────────────────────────────────────
    focal_alpha = 0.25
    focal_gamma = 2.0

    # ── Augmentation ──────────────────────────────────────────────────────────
    aug_rigid      = True
    aug_regularize = True
    aug_clahe      = True
    aug_ben_graham = True
    aug_intensity  = 0.7

    # ── CLAHE ─────────────────────────────────────────────────────────────────
    clahe_clip = 2.0
    clahe_grid = (4, 4)

    # ── ImageNet normalization (pretrained backbones) ─────────────────────────
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std  = [0.229, 0.224, 0.225]


# =============================================================================
# CustomConfig  —  SEResNet9, train everything from scratch
# =============================================================================
class CustomConfig(Config):
    pretrained          = False
    aug_intensity       = 0.9
    unfreeze_layers     = -1        # train all parameters
    num_epochs          = 60
    learning_rate       = 5e-4
    weight_decay        = 5e-4
    lr_patience         = 5
    early_stopping      = 15
    batch_size          = 16
    focal_alpha         = 0.5       # balanced (pairs with WeightedRandomSampler)
    warmup_epochs       = 5
    warmup_start_factor = 0.01

    checkpoint_dir  = Config.base_dir / "results" / "checkpoints" / "custom"
    checkpoint_name = "seresnet9_best.pth"
    output_path     = Config.output_dir / "output_custom.csv"


# =============================================================================
# FineTuneConfig  —  EnsembleModel, ImageNet pretrained + partial fine-tuning
# =============================================================================
class FineTuneConfig(Config):
    pretrained      = True
    aug_intensity   = 0.6
    unfreeze_layers = 3
    num_epochs      = 50
    learning_rate   = 3e-4
    weight_decay    = 1e-4
    lr_patience     = 3
    early_stopping  = 10
    batch_size      = 16

    checkpoint_dir  = Config.base_dir / "results" / "checkpoints" / "finetune"
    checkpoint_name = "multi_resolution_ensemble.pth"
    output_path     = Config.output_dir / "output_ft.csv"
