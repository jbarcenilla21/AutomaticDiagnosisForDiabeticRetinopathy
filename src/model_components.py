# =============================================================================
# model_components.py  —  Reusable building blocks for DR classification
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """Small CNN trained from scratch for binary DR classification.

    Args:
        img_size: Spatial size of the (square) input image in pixels.
                  Defaults to 224 to match the EfficientNet/DenseNet pipeline.
    ---
    Usage:
        from src.model_components import CustomCNN
        model = CustomCNN(img_size=224)
        logit = model(torch.zeros(2, 3, 224, 224))  # shape (2, 1)
    """

    def __init__(self, img_size: int = 224):
        super().__init__()

        # ── Feature extractor ─────────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, padding=0, bias=True),   # 3→16 (224x224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=True),  # 16→32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Regularisation
            nn.Dropout(p=0.25),
        )

        # ── Compute flat dimension for FC layer via dummy forward pass ─────────
        with torch.no_grad():
            dummy      = torch.zeros(1, 3, img_size, img_size)
            flat_size  = self.features(dummy).view(1, -1).size(1)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1, bias=True),    # raw logit — no sigmoid here
        )

        # ── Weight initialisation (He for ReLU paths) ─────────────────────────
        self._init_weights()

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[CustomCNN] img_size={img_size} | flat={flat_size:,} | "
            f"total={total:,}  trainable={trainable:,}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (B, 3, H, W).
        Returns:
            Raw logit tensor of shape (B, 1).
        """
        x = self.features(x)
        return self.classifier(x)