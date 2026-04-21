# =============================================================================
# config.py  —  Global configuration for DR classification pipeline
# =============================================================================

"""
Three configuration classes are provided:
  Config         — shared base defaults
  CustomConfig   — from-scratch training (no pretrained weights)
  FineTuneConfig — transfer learning from ImageNet pretrained models
 
Use the appropriate subclass in each training section of main.ipynb.
"""

import torch
from pathlib import Path
from torch.cuda import is_available
from colorama import Fore


class Config:

    # ---- Hardware & paths ----
    device = "cuda" if is_available() else "cpu"

    base_dir        = Path(__file__).resolve().parent.parent
    checkpoint_dir  = base_dir / "results" / "checkpoints"
    results_dir     = base_dir / "results"
    output_dir      = base_dir / "results" / "outputs"
    train_csv       = base_dir / "data" / "train.csv"
    val_csv         = base_dir / "data" / "val.csv" 
    test_csv        = base_dir / "data" / "test.csv"    
    data_dir        = base_dir / "data"
    images_dir      = base_dir / "data" / "images"


    # ---- Data parameters ----
    img_height   = 224          # EfficientNet / DenseNet / ViT-compatible
    img_width    = 224
    num_channels = 3
    rescale_size = 256          # rescale so random/center crop always fits

    # ---- Training hyperparameters ----
    num_clasess       = 1       # binary → single logit, BCEWithLogitsLoss
    unfreeze_layers   = 2       # how many layer groups to unfreeze in BaseModel
    batch_size      = 32
    num_epochs      = 50
    learning_rate   = 3e-4      # AdamW starting LR
    weight_decay    = 1e-4
    lr_patience     = 3         # ReduceLROnPlateau patience (epochs)
    lr_factor       = 0.3       # LR multiplier on plateau
    early_stopping  = 10        # early-stopping patience (val AUC epochs)
    # PIN_MEMORY      = DEVICE.type == "cuda"
    seed = 42

    # ---- Augmentation hyperparameters ----
    aug_rigid       = True      # rotation, h/v flip
    aug_regularize  = True      # random crop, Gaussian noise
    aug_clahe       = True      # CLAHE on green channel (medical-specific)
    aug_intensity   = 0.7       # intesity of augmentation [0.0 → no aug, 1.0 → maximum aug]

    # ---- CLAHE parameters ----
    clahe_clip      = 2.0       # clipLimit for cv2.createCLAHE
    clahe_grid      = (4, 4)    # tileGridSize for cv2.createCLAHE

    # ImageNet normalisation (used for pretrained backbones) 
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std  = [0.229, 0.224, 0.225]

    # ---- Cosmetics ----
    bar_format = (
        f"{Fore.CYAN}{{l_bar}}"
        f"{Fore.GREEN}{{bar:30}}"
        f"{Fore.YELLOW}{{n_fmt}}/{Fore.YELLOW}{{total_fmt}} "
        f"{Fore.MAGENTA}[{{elapsed}}<{Fore.MAGENTA}{{remaining}}] "
        f"{Fore.BLUE}{{rate_fmt}}"
    )


# =============================================================================
# CustomConfig
# =============================================================================
class CustomConfig(Config):
    pretrained      = False
    aug_intensity   = 0.9       # aggressive augmentation for random-init models
    unfreeze_layers = -1        # train everything from scratch
    num_epochs      = 60
    learning_rate   = 5e-4     # higher LR for random init
    weight_decay    = 5e-4
    lr_patience     = 5
    early_stopping  = 15
    batch_size      = 32
 
    checkpoint_dir  = Config.base_dir / "results" / "checkpoints" / "custom"
    output_path     = Config.output_dir / "output_custom.csv"
 

# =============================================================================
# FineTuneConfig
# =============================================================================
class FineTuneConfig(Config):
    pretrained      = True
    aug_intensity   = 0.6
    unfreeze_layers = 2
    num_epochs      = 50
    learning_rate   = 3e-4
    weight_decay    = 1e-4
    lr_patience     = 3
    early_stopping  = 10
    batch_size      = 32
 
    checkpoint_dir  = Config.base_dir / "results" / "checkpoints" / "finetune"
    output_path     = Config.output_dir / "output_ft.csv"