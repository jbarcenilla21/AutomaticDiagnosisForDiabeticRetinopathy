# =============================================================================
# model.py  —  BaseModel (single fine-tuned backbone) + EnsembleModel
# =============================================================================
"""
Architecture choices
────────────────────
BaseModel
  Wraps any torchvision / timm model.  The last ``unfreeze_n`` layer groups are
  unfrozen for fine-tuning; the rest stay frozen.  The classification head is
  replaced with a single linear output (raw logit for BCEWithLogitsLoss).

EnsembleModel
  Combines three complementary architectures:
    • EfficientNet-B2  — compact, accurate, good on medical images
    • DenseNet-121     — dense feature reuse, strong gradient flow
    • TinyViT-21M      — vision transformer, captures global context

  Predictions are averaged at the soft-probability level (after sigmoid).
  Each backbone has its own head; the ensemble wrapper is parameter-free.

  Note: TinyViT requires the ``timm`` library (pip install timm).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print(
        "[model.py] WARNING: 'timm' not installed.  "
        "TinyViT will be replaced by a MobileNetV3 fallback.\n"
        "  Install with:  pip install timm"
    )

from utils.config import Config as cfg



# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_params(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"total={total:,}  trainable={trainable:,}"


def _replace_head(model: nn.Module, backbone_name: str, out_features: int = 1) -> nn.Module:
    """Replace the classification head of common torchvision architectures."""
    name = backbone_name.lower()

    if "efficientnet" in name:
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_feat, out_features),
        )
    elif "densenet" in name:
        in_feat = model.classifier.in_features
        model.classifier = nn.Linear(in_feat, out_features)
    elif "resnet" in name or "resnext" in name:
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, out_features)
    elif "mobilenet_v3" in name:
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feat, out_features)
    elif "vit" in name or "swin" in name:
        # timm models expose a unified head attribute
        in_feat = model.head.in_features
        model.head = nn.Linear(in_feat, out_features)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            "Add a head-replacement branch or subclass BaseModel."
        )
    return model


def _get_layer_groups(model: nn.Module, backbone_name: str) -> list[nn.Module]:
    """Return ordered layer groups (coarse → fine) for progressive unfreezing."""
    name = backbone_name.lower()
    if "efficientnet" in name:
        return [model.features[i] for i in range(len(model.features))] + [model.classifier]
    elif "densenet" in name:
        return [
            model.features.conv0,
            model.features.denseblock1,
            model.features.denseblock2,
            model.features.denseblock3,
            model.features.denseblock4,
            model.classifier,
        ]
    elif "mobilenet_v3" in name:
        return [model.features, model.classifier]
    elif "tiny_vit" in name:
        # TinyViT models in timm use 'stages' instead of 'blocks'
        return list(model.stages) + [model.head]
    elif "vit" in name or "swin" in name:
        # Standard ViT/Swin models in timm use 'blocks'
        return list(model.blocks) + [model.head]
    else:
        # generic fallback: treat children as groups
        return list(model.children())


# ─────────────────────────────────────────────────────────────────────────────
# BaseModel  —  single fine-tuned backbone
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(nn.Module):
    """Single pretrained backbone fine-tuned for binary DR classification.

    Args:
        backbone_name: One of ``efficientnet_b2``, ``densenet121``,
                       ``mobilenet_v3_large``, or any timm model name.
        unfreeze_n:    Number of layer groups (from the end) to leave trainable.
                       Set to ``-1`` to unfreeze the entire network.
        pretrained:    Load ImageNet weights.  False → random init (Custom track).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        unfreeze_n: int = cfg.unfreeze_layers,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.model = self._build_backbone(backbone_name, pretrained)
        self._freeze_and_unfreeze(unfreeze_n)
        print(f"[BaseModel] {backbone_name} | {_count_params(self.model)}")

    # ── Construction helpers ──────────────────────────────────────────────────

    def _build_backbone(self, name: str, pretrained: bool) -> nn.Module:
        weights_arg = "IMAGENET1K_V1" if pretrained else None
        name_lower  = name.lower()

        if name_lower == "efficientnet_b2":
            m = models.efficientnet_b2(weights=weights_arg)
        elif name_lower == "efficientnet_b3":
            m = models.efficientnet_b3(weights=weights_arg)
        elif name_lower == "densenet121":
            m = models.densenet121(weights=weights_arg)
        elif name_lower == "mobilenet_v3_large":
            m = models.mobilenet_v3_large(weights=weights_arg)
        elif TIMM_AVAILABLE:
            m = timm.create_model(name, pretrained=pretrained, num_classes=1)
            return m   # timm heads already replaced
        else:
            raise ValueError(
                f"Backbone '{name}' requires timm.  "
                "Install with:  pip install timm"
            )

        _replace_head(m, name_lower)
        return m

    def _freeze_and_unfreeze(self, unfreeze_n: int):
        if unfreeze_n == -1:
            return   # train everything

        # Freeze all first
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze the last unfreeze_n layer groups
        groups = _get_layer_groups(self.model, self.backbone_name)
        for group in groups[-unfreeze_n:]:
            for p in group.parameters():
                p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # raw logit → (B, 1)

    # ── Convenience ───────────────────────────────────────────────────────────

    def unfreeze_all(self):
        """Unfreeze the full network (useful after initial warm-up)."""
        for p in self.model.parameters():
            p.requires_grad = True
        print(f"[BaseModel] All layers unfrozen. {_count_params(self.model)}")


# ─────────────────────────────────────────────────────────────────────────────
# EnsembleModel  —  EfficientNet-B2 + DenseNet-121 + TinyViT
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleModel(nn.Module):
    """Soft-voting ensemble of three complementary CNN / ViT backbones.

    Each backbone produces a raw logit; sigmoid converts it to a probability.
    The ensemble output is the mean probability across all members.

    Architecture selection rationale
    ──────────────────────────────────
    EfficientNet-B2   Efficient compound scaling; very sample-efficient.
    DenseNet-121      Dense connectivity improves gradient flow on small datasets.
    TinyViT-21M       Global self-attention complements local CNN features.

    Args:
        unfreeze_n:  Layer groups to unfreeze per backbone.
        pretrained:  Load ImageNet weights for all members.
        weights:     Optional list [w0, w1, w2] for weighted averaging (uniform
                     averaging if None).
    """

    def __init__(
        self,
        unfreeze_n: int = cfg.unfreeze_layers,
        pretrained: bool = True,
        weights: list[float] | None = None,
    ):
        super().__init__()

        # ── Build each backbone ───────────────────────────────────────────────
        self.eff_net = BaseModel(
            backbone_name="efficientnet_b2",
            unfreeze_n=unfreeze_n,
            pretrained=pretrained,
        )
        self.dense_net = BaseModel(
            backbone_name="densenet121",
            unfreeze_n=unfreeze_n,
            pretrained=pretrained,
        )

        # TinyViT via timm; fall back to MobileNetV3 if timm is absent
        if TIMM_AVAILABLE:
            self.tiny_vit = BaseModel(
                backbone_name="tiny_vit_21m_224.dist_in22k_ft_in1k",
                unfreeze_n=unfreeze_n,
                pretrained=pretrained,
            )
        else:
            print("[EnsembleModel] Using MobileNetV3 as TinyViT fallback.")
            self.tiny_vit = BaseModel(
                backbone_name="mobilenet_v3_large",
                unfreeze_n=unfreeze_n,
                pretrained=pretrained,
            )

        # ── Averaging weights ─────────────────────────────────────────────────
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32)
            w = w / w.sum()
        else:
            w = torch.ones(3) / 3.0
        self.register_buffer("ens_weights", w)   # saved with state_dict

        total = (
            sum(p.numel() for p in self.parameters() if p.requires_grad)
        )
        print(f"[EnsembleModel] Total trainable params: {total:,}")

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return weighted-mean sigmoid probability. Shape: (B, 1)."""
        p0 = torch.sigmoid(self.eff_net(x))    # (B, 1)
        p1 = torch.sigmoid(self.dense_net(x))  # (B, 1)
        p2 = torch.sigmoid(self.tiny_vit(x))   # (B, 1)

        w = self.ens_weights  # (3,)
        out = w[0] * p0 + w[1] * p1 + w[2] * p2  # (B, 1)
        return out

    # ── Utilities ─────────────────────────────────────────────────────────────

    def unfreeze_all_members(self):
        """Unfreeze every parameter in every backbone (use after warm-up)."""
        for member in [self.eff_net, self.dense_net, self.tiny_vit]:
            member.unfreeze_all()
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EnsembleModel] All members unfrozen. Trainable: {total:,}")

    def members(self) -> list[BaseModel]:
        return [self.eff_net, self.dense_net, self.tiny_vit]

    def member_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return individual raw logits for per-backbone diagnostics."""
        return [self.eff_net(x), self.dense_net(x), self.tiny_vit(x)]