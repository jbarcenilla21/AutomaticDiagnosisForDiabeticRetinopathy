# =============================================================================
# config.py  —  Global configuration for DR classification pipeline
# =============================================================================

import os
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
    train_csv       = base_dir / "data" / "train.csv"
    val_csv         = base_dir / "data" / "val.csv" 
    test_csv        = base_dir / "data" / "test.csv"    
    data_dir        = base_dir / "data"
    images_dir      = base_dir / "data" / "images"
    output_path     = base_dir / "results" / "outputs"


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
