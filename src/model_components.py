# =============================================================================
# model_components.py  —  Simplified VGG Architecture + Focal Loss
# =============================================================================

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CustomVGG  —  Lightweight VGG-style CNN (Custom track)
# =============================================================================

class CustomVGG(nn.Module):
    """Simplified VGG-style CNN for DR binary classification.

    Uses stacked 3×3 convolutions (VGG philosophy) to increase receptive field
    depth.  MaxPool2d halves spatial resolution between blocks.  The head
    outputs a single raw logit (no sigmoid) for BCEWithLogitsLoss / FocalLoss.

    Args:
        img_size:    Expected input spatial size (height == width).
        num_classes: Number of output neurons.  Always 1 for binary DR.
    """

    def __init__(self, img_size: int = 512, num_classes: int = 1):
        super().__init__()

        # ── Feature extractor (VGG-style blocks) ──────────────────────────────
        # padding=1 preserves spatial size within each block.
        # MaxPool2d(2, 2) halves it between blocks.
        self.features = nn.Sequential(
            # Block 1: img_size → img_size/2
            self._make_vgg_block(3,   16,  num_convs=2),
            # Block 2: img_size/2 → img_size/4
            self._make_vgg_block(16,  32,  num_convs=2),
            # Block 3: img_size/4 → img_size/8
            self._make_vgg_block(32,  64, num_convs=3),
            # Block 4: img_size/8 → img_size/16
            self._make_vgg_block(64,  128, num_convs=3),
            # Adaptive pool → always (B, 128, 7, 7) regardless of img_size
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        # ── Compute flat dimension dynamically ────────────────────────────────
        with torch.no_grad():
            dummy     = torch.zeros(1, 3, img_size, img_size)
            flat_size = self.features(dummy).view(1, -1).size(1)

        # ── Classifier head ───────────────────────────────────────────────────
        # Returns a raw logit — sigmoid is applied externally by the loss.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),   # raw logit
        )

        self._init_weights()

        total = sum(p.numel() for p in self.parameters())
        print(f"[CustomVGG] img_size={img_size} | flat={flat_size:,} | total_params={total:,}")

    # ── Helper builders ───────────────────────────────────────────────────────

    def _make_vgg_block(
        self, in_channels: int, out_channels: int, num_convs: int
    ) -> nn.Sequential:
        """Create a VGG-style block of stacked 3×3 conv layers + MaxPool."""
        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """He (Kaiming) init for Conv2d; constant init for BatchNorm."""
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

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 3, H, W).
        Returns:
            Raw logit tensor (B, 1).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

class SimpleLeNet(nn.Module):
    """Very small CNN inspired by LeNet-5, adapted for RGB medical images."""
    def __init__(self, img_size: int = 512, num_classes: int = 1):
        super().__init__()

        # ── Feature extractor ────────────────────────────────────────────────
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),   # /2

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),   # /4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # /8

            nn.AdaptiveAvgPool2d((2, 2))  # keep VERY small
        )

        # ── Compute flattened size ───────────────────────────────────────────
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            flat_size = self.features(dummy).view(1, -1).size(1)

        # ── Classifier (tiny) ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[SimpleLeNet] params={total:,}")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =============================================================================
# FocalLoss  —  Binary Focal Loss (handles both logits and probabilities)
# =============================================================================

class FocalLoss(nn.Module):
    """Binary Focal Loss for imbalanced classification.

    The focal term ``(1 − p_t)^gamma`` down-weights easy examples (large p_t)
    so the network focuses on hard positives (small lesions, borderline cases).

    Formula:  FL(p_t) = −α · (1 − p_t)^γ · log(p_t)

    Args:
        alpha:       Class-balance weight ∈ [0, 1].  0.25 means the positive
                     class receives 0.25× the base weight.
        gamma:       Focusing exponent ≥ 0.  Higher = more focus on hard examples.
                     gamma=0 reduces to standard BCE (no focusing).
        from_logits: If True  → inputs are raw logits  (use with BaseModel / CustomVGG).
                     If False → inputs are probabilities (use with EnsembleModel,
                                which applies sigmoid internally).
    """

    def __init__(
        self,
        alpha:       float = 0.25,
        gamma:       float = 2.0,
        from_logits: bool  = True,
    ):
        super().__init__()
        self.alpha       = alpha
        self.gamma       = gamma
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs:  Logits (B,) or (B,1) if from_logits=True;
                     Probabilities (B,) or (B,1) if from_logits=False.
            targets: Binary labels (B,) in {0, 1}.
        Returns:
            Scalar focal loss.
        """
        targets = targets.view(-1, 1).float()

        if self.from_logits:
            # Numerically stable: compute BCE from logits, derive probs after
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
            probs = torch.sigmoid(inputs)
        else:
            # Inputs are already probabilities — use standard BCE
            probs    = inputs.clamp(1e-7, 1.0 - 1e-7)   # avoid log(0)
            bce_loss = F.binary_cross_entropy(probs, targets, reduction="none")

        # p_t = predicted probability of the TRUE class
        pt          = torch.where(targets == 1, probs, 1.0 - probs)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma

        return (focal_weight * bce_loss).mean()