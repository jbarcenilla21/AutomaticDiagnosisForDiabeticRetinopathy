"""Custom CNNs built from scratch for binary DR classification.

No pretrained weights — compliant with the Custom track rules.
All models output raw logits (N, 1). Use BCEWithLogitsLoss or FocalLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _ResBlock(nn.Module):
    """Two Conv3×3+BN+ReLU layers with an identity skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_relu(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class _SEBlock(nn.Module):
    """Channel-wise attention: suppresses noise channels, amplifies lesion channels."""

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
# SEResNet9  — primary Custom track model (val AUC 0.7292)
# ─────────────────────────────────────────────────────────────────────────────
# Spatial flow at 512×512:
#   prep:    3→32  ch   512×512
#   layer1:  32→64 ch   256×256  (MaxPool)
#   res1 + SE(64)        256×256
#   layer2:  64→128 ch  128×128  (MaxPool)
#   layer3: 128→256 ch   64×64   (MaxPool)
#   res2 + SE(256)        64×64
#   GAP → Dropout(0.1) → Linear(256, 1)   ~1.8M params

class SEResNet9(nn.Module):
    """ResNet-9 with Squeeze-and-Excitation attention blocks.

    Key improvements over plain ResNet9:
    - SE blocks for learned channel re-weighting after each ResBlock
    - Extra 128→256 downsampling stage (capacity ~1.8M params)
    - dropout=0.1 (raise to 0.3 once train AUC > 0.85)
    - Kaiming (He) normal initialization on all Conv2d layers
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.1):
        super().__init__()

        self.prep   = _conv_bn_relu(in_channels, 32, kernel_size=3, padding=1)

        self.layer1 = nn.Sequential(
            _conv_bn_relu(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.res1   = _ResBlock(64)
        self.se1    = _SEBlock(64)

        self.layer2 = nn.Sequential(
            _conv_bn_relu(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            _conv_bn_relu(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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


# ─────────────────────────────────────────────────────────────────────────────
# CustomVGG  —  Lightweight VGG-style CNN (used by EnsembleModel as "customvgg")
# ─────────────────────────────────────────────────────────────────────────────

class CustomVGG(nn.Module):
    """Simplified VGG-style CNN for DR binary classification.

    Outputs a single raw logit (no sigmoid) for BCEWithLogitsLoss / FocalLoss.
    """

    def __init__(self, img_size: int = 512, num_classes: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            self._make_vgg_block(3,   16,  num_convs=2),
            self._make_vgg_block(16,  32,  num_convs=2),
            self._make_vgg_block(32,  64,  num_convs=3),
            self._make_vgg_block(64,  128, num_convs=3),
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        with torch.no_grad():
            dummy     = torch.zeros(1, 3, img_size, img_size)
            flat_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        print(f"[CustomVGG] img_size={img_size} | flat={flat_size:,} | total_params={total:,}")

    def _make_vgg_block(self, in_channels: int, out_channels: int, num_convs: int) -> nn.Sequential:
        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SimpleLeNet  —  Tiny baseline CNN
# ─────────────────────────────────────────────────────────────────────────────

class SimpleLeNet(nn.Module):
    """Very small CNN inspired by LeNet-5, adapted for RGB medical images."""

    def __init__(self, img_size: int = 512, num_classes: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            flat_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[SimpleLeNet] params={total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
