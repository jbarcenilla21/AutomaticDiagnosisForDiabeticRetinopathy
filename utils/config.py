# =============================================================================
# config.py  —  Global configuration for DR classification pipeline
# =============================================================================
"""
Three configuration classes are provided:
  Config         — shared base defaults
  CustomConfig   — from-scratch training (no pretrained weights)
  FineTuneConfig — transfer learning from ImageNet pretrained models

Use the appropriate subclass in each training section of main_combined.ipynb.
"""

from pathlib import Path
from torch.cuda import is_available


class Config:

    # ── Hardware & paths ──────────────────────────────────────────────────────
    device = "cuda" if is_available() else "cpu"

    base_dir        = Path(__file__).resolve().parent.parent
    checkpoint_name = Path("best_model.pth")  # default checkpoint filename
    checkpoint_dir  = base_dir / "results" / "checkpoints"
    results_dir     = base_dir / "results"
    output_dir      = base_dir / "results" / "outputs"
    train_csv       = base_dir / "data" / "train.csv"
    val_csv         = base_dir / "data" / "val.csv"
    test_csv        = base_dir / "data" / "test.csv"
    data_dir        = base_dir / "data"
    images_dir      = base_dir / "data" / "images"

    # ── Multi-scale ensemble resolution (one scale per member) ───────────────
    # Model A: 224×224 — global context  (low-res, broad features)
    # Model B: 384×384 — mid-scale       (balanced detail/context)
    # Model C: 512×512 — fine lesions    (full resolution, small structures)
    ensemble_scales = [224, 384, 512]

    # ── Data parameters ───────────────────────────────────────────────────────
    # Use 512 as the canonical input size (largest ensemble scale).
    # rescale_size must be slightly larger than img_height so that
    # RandomCrop always has room to crop without padding artifacts.
    img_height   = 512
    img_width    = 512
    num_channels = 3
    rescale_size = 580          # rescale shortest side before RandomCrop(512)

    # ── Training hyperparameters ──────────────────────────────────────────────
    num_clasess     = 1         # binary → single logit, BCEWithLogitsLoss
    unfreeze_layers = 2         # how many layer groups to unfreeze in BaseModel
    batch_size      = 16        # reduced for 512×512 inputs
    num_epochs      = 50
    learning_rate   = 3e-4     # AdamW starting LR
    weight_decay    = 1e-4
    lr_patience     = 3         # ReduceLROnPlateau patience (epochs)
    lr_factor       = 0.3       # LR multiplier on plateau
    early_stopping  = 10        # early-stopping patience (val AUC epochs)
    seed            = 42

    # ── TTA (Test-Time Augmentation) ──────────────────────────────────────────
    # Each test image is forward-passed num_tta times with random flips/rotations.
    # The final score is the mean over all passes.  Set to 1 to disable TTA.
    num_tta = 10

    # ── Focal Loss hyperparameters ────────────────────────────────────────────
    # alpha: balances positive/negative class weight in the focal formula.
    # gamma: focusing parameter — higher values down-weight easy examples more.
    focal_alpha = 0.25
    focal_gamma = 2.0

    # ── Augmentation flags & intensity ────────────────────────────────────────
    aug_rigid      = True       # rotation, h/v flip
    aug_regularize = True       # Gaussian noise
    aug_clahe      = True       # dual CLAHE (green + red channels)
    aug_ben_graham = True       # alias kept for backward compatibility
    aug_intensity  = 0.7        # global magnitude scalar [0.0, 1.0]

    # ── CLAHE parameters ──────────────────────────────────────────────────────
    clahe_clip = 2.0            # clipLimit for cv2.createCLAHE
    clahe_grid = (4, 4)         # tileGridSize for cv2.createCLAHE

    # ── ImageNet normalization (used for pretrained backbones) ────────────────
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std  = [0.229, 0.224, 0.225]


# =============================================================================
# CustomConfig  —  train everything from scratch
# =============================================================================
class CustomConfig(Config):
    pretrained      = False
    aug_intensity   = 0.9       # aggressive augmentation for random-init models
    unfreeze_layers = -1        # -1 → train all parameters (no freezing)
    num_epochs      = 60
    learning_rate   = 5e-4     # higher LR for random initialisation
    weight_decay    = 5e-4
    lr_patience     = 5
    early_stopping  = 15
    batch_size      = 16

    checkpoint_dir  = Config.base_dir / "results" / "checkpoints" / "custom"
    checkpoint_name = "multiRes_custom_ensemble.pth"
    output_path     = Config.output_dir / "output_custom.csv"


# =============================================================================
# FineTuneConfig  —  ImageNet pretrained + partial fine-tuning
# =============================================================================
class FineTuneConfig(Config):
    pretrained      = True
    aug_intensity   = 0.6
    unfreeze_layers = 3         # unfreeze the last 3 layer groups
    num_epochs      = 50
    learning_rate   = 3e-4
    weight_decay    = 1e-4
    lr_patience     = 3
    early_stopping  = 10
    batch_size      = 16        # 512×512 needs smaller batches than 224×224

    checkpoint_dir  = Config.base_dir / "results" / "checkpoints" / "finetune"
    checkpoint_name = "multi_resolution_ensemble.pth"
    output_path     = Config.output_dir / "output_ft.csv"