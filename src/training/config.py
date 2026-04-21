# Training hyperparameters

# ── Image dimensions ──────────────────────────────────────────────────────────
img_height   = 224
img_width    = 224
num_channels = 3

# ── Fine-tune / legacy CustomNet (SGD) ───────────────────────────────────────
learning_rate = 1e-2
batch_size    = 32
num_epochs    = 50
lr_step_size  = 10
lr_gamma      = 0.1
momentum      = 0.9

# ── EnsembleNet: DenseNetSmall + ResNet-9 (Adam) ─────────────────────────────
# Green-channel split happens inside the model; no change to the data pipeline.
ensemble_lr           = 5e-4   # Adam default; safe for both sub-models
ensemble_weight_decay = 1e-4   # mild L2 regularisation
ensemble_batch_size   = 32     # fits Colab Pro T4/V100 comfortably
ensemble_num_epochs   = 60     # early stopping in trainer caps the actual run
ensemble_lr_step_size = 15     # decay at epoch 15 and 30
ensemble_lr_gamma     = 0.1
