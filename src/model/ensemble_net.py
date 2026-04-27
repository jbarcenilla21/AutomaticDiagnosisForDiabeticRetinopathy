"""Ensemble: DenseNetSmall (green channel) + ResNet-9 (full RGB).

Custom track — no pretrained weights.

Architecture rationale:
  - DenseNetSmall receives only the green channel (index 1 of RGB).
    The green channel has the highest contrast for microaneurysms, haemorrhages,
    and retinal vessels — the primary lesion markers in DR.
  - ResNet9 receives the full RGB image, capturing colour cues such as exudate
    yellow and vessel redness that the green channel discards.
  - The two models have different receptive-field profiles (dense reuse vs
    skip-connection shortcuts), so their errors are largely uncorrelated.
  - Final score = mean of the two sigmoid probabilities → BCELoss in trainer.

Input:  (N, 3, H, W)  float32 tensor (normalised with ImageNet stats).
Output: (N, 1)        float32 tensor of DR probabilities in [0, 1].
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DenseNet-small  (operates on the green channel — 1-channel input)
# ─────────────────────────────────────────────────────────────────────────────
# Spatial flow on 224×224:
#   Stem: Conv3×3 + MaxPool(2)  →  112×112
#   Block1 (4 layers, k=12): 32 → 80 ch  |  112×112
#   Transition1 (×0.5):  80 → 40 ch      |   56×56
#   Block2 (4 layers, k=12): 40 → 88 ch  |   56×56
#   Transition2 (×0.5):  88 → 44 ch      |   28×28
#   Block3 (4 layers, k=12): 44 → 92 ch  |   28×28
#   GAP → Dropout → Linear(92, 1)

class _DenseLayer(nn.Module):
    """BN → ReLU → Conv3×3 (no bottleneck, keeps code simple)."""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                      padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.layer(x)], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, n_layers: int, in_channels: int, growth_rate: int):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(n_layers):
            layers.append(_DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Transition(nn.Module):
    """1×1 conv (compression) + 2×2 average pooling."""

    def __init__(self, in_channels: int, compression: float = 0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, 2),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DenseNetSmall(nn.Module):
    """Lightweight DenseNet for the green channel (1 input channel).

    ~90 k trainable parameters.
    """

    def __init__(self, in_channels: int = 1, growth_rate: int = 12,
                 block_layers: tuple = (4, 4, 4)):
        super().__init__()

        # Stem
        stem_ch = 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112
        )

        # Dense blocks + transitions
        ch = stem_ch
        self.block1 = _DenseBlock(block_layers[0], ch, growth_rate)
        ch = self.block1.out_channels    # 32 + 4*12 = 80

        self.trans1 = _Transition(ch)
        ch = self.trans1.out_channels    # 40

        self.block2 = _DenseBlock(block_layers[1], ch, growth_rate)
        ch = self.block2.out_channels    # 40 + 4*12 = 88

        self.trans2 = _Transition(ch)
        ch = self.trans2.out_channels    # 44

        self.block3 = _DenseBlock(block_layers[2], ch, growth_rate)
        ch = self.block3.out_channels    # 44 + 4*12 = 92

        self.final_bn = nn.BatchNorm2d(ch)
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(ch, 1),            # raw logit
        )
        self._out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = torch.relu(self.final_bn(x))
        x = self.gap(x).flatten(1)
        return self.head(x)              # (N, 1) logit


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-9  (operates on full RGB — 3-channel input)
# ─────────────────────────────────────────────────────────────────────────────
# Spatial flow on 224×224:
#   Prep  : Conv3×3  →  32 ch  |  224×224
#   Layer1: Conv3×3 + MaxPool  →  64 ch  |  112×112
#   Res1  : ResBlock(64)                 |  112×112
#   Layer2: Conv3×3 + MaxPool  → 128 ch  |   56×56
#   Layer3: Conv3×3 + MaxPool  → 128 ch  |   28×28
#   Res2  : ResBlock(128)                |   28×28
#   GAP   → Dropout(0.5) → Linear(128,1)

class _ResBlock(nn.Module):
    """Two Conv3×3+BN+ReLU layers with an identity skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_relu(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


# ─────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation block
# ─────────────────────────────────────────────────────────────────────────────

class _SEBlock(nn.Module):
    """Channel-wise attention: suppresses noise channels, amplifies lesion channels.

    Args:
        channels:  number of input/output feature channels.
        reduction: bottleneck ratio for the excitation MLP.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.excitation(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# SEResNet9  — ResNet-9 with SE attention, increased capacity, Kaiming init
# ─────────────────────────────────────────────────────────────────────────────
# Spatial flow at 512×512:
#   prep:    3→32  ch   512×512
#   layer1:  32→64 ch   256×256  (MaxPool)
#   res1 + SE(64)        256×256
#   layer2:  64→128 ch  128×128  (MaxPool)
#   layer3: 128→256 ch   64×64   (MaxPool)
#   res2 + SE(256)        64×64
#   GAP → Dropout(0.1) → Linear(256, 1)
#
# ~1.8M trainable parameters.
# Returns raw logits — use BCEWithLogitsLoss or FocalLoss, not BCELoss.

class SEResNet9(nn.Module):
    """ResNet-9 with Squeeze-and-Excitation attention blocks.

    Key improvements over ResNet9:
    - SE blocks after each ResBlock for learned channel re-weighting
    - Extra 128->256 downsampling stage (increases capacity to ~1.8M params)
    - dropout=0.1 (underfitting phase; raise to 0.3 once train AUC > 0.80)
    - Kaiming (He) normal initialization on all Conv2d layers
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.1):
        super().__init__()

        self.prep   = _conv_bn_relu(in_channels, 32, kernel_size=3, padding=1)

        self.layer1 = nn.Sequential(
            _conv_bn_relu(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 512 -> 256
        )
        self.res1   = _ResBlock(64)
        self.se1    = _SEBlock(64)

        self.layer2 = nn.Sequential(
            _conv_bn_relu(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 256 -> 128
        )
        self.layer3 = nn.Sequential(
            _conv_bn_relu(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 128 -> 64
        )
        self.res2   = _ResBlock(256)
        self.se2    = _SEBlock(256)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.se1(self.res1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.se2(self.res2(x))
        x = self.gap(x).flatten(1)
        return self.head(x)              # (N, 1) raw logit


class ResNet9(nn.Module):
    """ResNet-9 adapted for 224×224 input with ~400 k trainable parameters."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.prep   = _conv_bn_relu(in_channels, 32, kernel_size=3, padding=1)

        self.layer1 = nn.Sequential(
            _conv_bn_relu(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 224 → 112
        )
        self.res1   = _ResBlock(64)

        self.layer2 = nn.Sequential(
            _conv_bn_relu(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 112 → 56
        )
        self.layer3 = nn.Sequential(
            _conv_bn_relu(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),          # 56 → 28
        )
        self.res2   = _ResBlock(128)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 1),           # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.res1(self.layer1(x))
        x = self.layer2(x)
        x = self.res2(self.layer3(x))
        x = self.gap(x).flatten(1)
        return self.head(x)              # (N, 1) logit


# ─────────────────────────────────────────────────────────────────────────────
# EnsembleNet — combines both branches
# ─────────────────────────────────────────────────────────────────────────────

class DenseNetGreen(nn.Module):
    """DenseNetSmall wrapper that accepts 3-channel input and uses only the green channel.

    Allows using the same 3-channel DataLoader for both base models.
    Outputs a raw logit — use BCEWithLogitsLoss, not BCELoss.
    """

    def __init__(self):
        super().__init__()
        self.backbone = DenseNetSmall(in_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x[:, 1:2, :, :])             # (N, 1) logit


class EnsembleNet(nn.Module):
    """DenseNetSmall (green channel) + ResNet9 (RGB) probability ensemble.

    Each sub-model outputs a raw logit. Sigmoid is applied to each logit and
    the two probabilities are averaged. The result is in [0, 1], compatible
    with nn.BCELoss in the trainer.
    """

    def __init__(self):
        super().__init__()
        self.densenet = DenseNetSmall(in_channels=1)
        self.resnet9  = ResNet9(in_channels=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        green = x[:, 1:2, :, :]                          # (N, 1, H, W)
        prob_d = torch.sigmoid(self.densenet(green))      # (N, 1)
        prob_r = torch.sigmoid(self.resnet9(x))           # (N, 1)
        return (prob_d + prob_r) / 2                      # (N, 1) ∈ [0, 1]
