# =============================================================================
# model_components.py  —  Simplified VGG Architecture for DR classification
# =============================================================================

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_VGG(nn.Module):
    """
    Simplified VGG implementation.
    Uses blocks of multiple 3x3 convolutions to increase receptive field depth,
    following the VGG-style design philosophy.
    """

    def __init__(self, img_size: int = 224, num_classes: int = 1):
        super().__init__()

        # ── Feature extractor (VGG Style) ─────────────────────────────────────
        # Standard VGG uses padding=1 to preserve spatial resolution within blocks,
        # reducing it only via MaxPool2d layers.
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            self._make_vgg_block(3, 32, num_convs=2),
            
            # Block 2: 112x112 -> 56x56
            self._make_vgg_block(32, 64, num_convs=2),
            
            # Block 3: 56x56 -> 28x28
            self._make_vgg_block(64, 128, num_convs=3),
            
            # Block 4: 28x28 -> 14x14
            self._make_vgg_block(128, 256, num_convs=3),
            
            # Global Average Pooling ensures a fixed size before the FC layers
            nn.AdaptiveAvgPool2d((7, 7)) 
        )

        # ── Compute flat dimension ────────────────────────────────────────────
        # With AdaptiveAvgPool2d(7,7), output size is always (Channels * 7 * 7)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            flat_size = self.features(dummy).view(1, -1).size(1)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes), # Raw logit output
        )

        self._init_weights()
        
        # Parameter logging
        total = sum(p.numel() for p in self.parameters())
        print(f"[Custom_VGG] img_size={img_size} | flat={flat_size:,} | total_params={total:,}")

    def _make_vgg_block(self, in_channels: int, out_channels: int, num_convs: int) -> nn.Sequential:
        """Helper to create blocks of multiple 3x3 convolutional layers."""
        layers = []
        for i in range(num_convs):
            conv_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels)) 
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initializes weights using He (Kaiming) normalization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 3, H, W).
        Returns:
            Logit tensor of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for unbalance classification.
    Formula: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Ensure targets have the same shape as inputs (B, 1)
        targets = targets.view(-1, 1).float()
        
        # Calculate the standard BCE with logits for stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt (the probability of the correct class)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Apply the focal loss modulation factor
        # (1 - pt)^gamma reduces the loss for easy examples (where pt is high)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        
        return focal_loss.mean()