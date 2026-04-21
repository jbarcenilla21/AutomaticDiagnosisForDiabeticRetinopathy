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

# ── EnsembleNet: DenseNetSmall + ResNet-9 (Adam + AMP) ───────────────────────
# Tuned for A100 / H100 (≥80 GB GPU RAM, ≥100 GB system RAM).
ensemble_lr           = 1e-3   # higher LR justified by larger batch
ensemble_weight_decay = 1e-4
ensemble_batch_size   = 64     # comfortably fits 512×512 on 80 GB GPU
ensemble_val_batch    = 128    # no gradients → can double the batch
ensemble_num_epochs   = 60
ensemble_cosine_tmax  = 60     # CosineAnnealingLR period = full training run
ensemble_cosine_etamin = 1e-6  # minimum LR at the end of the cosine cycle
ensemble_num_workers  = 4      # 4 CPU workers saturate A100 data pipeline
use_amp               = True   # mixed-precision: ~2-3× speedup on A100
