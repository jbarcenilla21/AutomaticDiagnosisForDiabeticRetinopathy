# Training hyperparameters

# ── Image dimensions ──────────────────────────────────────────────────────────
# 512×512: 4× more pixels than 224 — critical for detecting small DR lesions
# (microaneurysms, haemorrhages) that get lost at lower resolutions.
img_height   = 512
img_width    = 512
num_channels = 3

# ── Fine-tune / legacy CustomNet (SGD) ───────────────────────────────────────
learning_rate = 1e-2
batch_size    = 32
num_epochs    = 50
lr_step_size  = 10
lr_gamma      = 0.1
momentum      = 0.9

# ── Class imbalance ──────────────────────────────────────────────────────────
# Train set: 1468 negatives / 532 positives → ratio 2.759.
# Used as pos_weight in BCEWithLogitsLoss so each positive sample weighs 2.76×.
pos_weight = 2.759

# ── Base models: DenseNetGreen + ResNet9 (Adam + AMP) ────────────────────────
# Tuned for A100 / H100 (≥80 GB GPU RAM, ≥100 GB system RAM).
base_lr            = 1e-3
base_weight_decay  = 1e-4
base_batch_size    = 64     # comfortably fits 512×512 on 80 GB GPU
base_val_batch     = 128    # no gradients → can double the batch
base_num_epochs    = 60
base_cosine_tmax   = 60     # CosineAnnealingLR period = full training run
base_cosine_etamin = 1e-6

# ── LR warmup ────────────────────────────────────────────────────────────────
# Start at lr * 0.01 and scale linearly to lr over the first 5 epochs before
# handing off to CosineAnnealingLR. Prevents gradient explosion at random init.
warmup_epochs       = 5
warmup_start_factor = 0.01

base_num_workers   = 4
use_amp            = True   # mixed-precision: ~2-3× speedup on A100

# Legacy names kept for fine-tune track
ensemble_lr           = base_lr
ensemble_weight_decay = base_weight_decay
ensemble_batch_size   = base_batch_size
ensemble_val_batch    = base_val_batch
ensemble_num_epochs   = base_num_epochs
ensemble_cosine_tmax  = base_cosine_tmax
ensemble_cosine_etamin = base_cosine_etamin
ensemble_num_workers  = base_num_workers
