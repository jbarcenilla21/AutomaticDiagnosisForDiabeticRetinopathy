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
    Supported backbones (torchvision):
    efficientnet_b2, efficientnet_b3
    densenet121
    mobilenet_v3_large
    vgg16, vgg16_bn, vgg19, vgg19_bn
    resnet50, resnext50_32x4d
 
    Any timm model is also supported if timm is installed.

EnsembleModel
    Heterogeneous soft-voting ensemble.  Accepts a list of:
        • Backbone name strings  →  wrapped with BaseModel automatically.
        • "CustomCNN"            →  instantiates CustomCNN from model_components.
        • Pre-built nn.Module    →  used as-is (head already replaced externally).
    
    All members are assumed to output raw logits.  The ensemble wrapper applies
    sigmoid to each member's logit and returns the weighted-mean probability.
    Trainer detects EnsembleModel and switches to BCELoss automatically.
    
    Default backbone set (backward-compatible):
        EfficientNet-B2 + DenseNet-121 + TinyViT-21M-224
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
    elif "vgg" in name:
        in_feat = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_feat, out_features)
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
    elif "vgg" in name:
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
        elif name_lower == "vgg16":
            m = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
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

    Args:
        backbone_names: List whose elements can be:
                        • A backbone name string accepted by BaseModel
                          (e.g. ``"efficientnet_b2"``, ``"vgg16_bn"``).
                        • The special string ``"CustomCNN"`` to auto-instantiate
                          the lightweight custom architecture.
                        • A pre-built ``nn.Module`` instance whose forward()
                          returns a raw logit of shape ``(B, 1)``.
                        Defaults to the original three-member ensemble.
        unfreeze_n:     Layer groups to unfreeze per string-based backbone member.
                        Forced to ``-1`` when ``pretrained=False``.
        pretrained:     Load ImageNet weights for all string-based members.
        weights:        Optional ``[w0, w1, …]`` for weighted averaging.
                        Uniform weighting when ``None``.
 
    Forward returns:
        Weighted-mean sigmoid probability, shape ``(B, 1)``.
    """

    _DEFAULT_BACKBONES = [
        "efficientnet_b2",
        "densenet121",
        "tiny_vit_21m_224.dist_in22k_ft_in1k",
    ]
 
    def __init__(
        self,
        backbone_names: list | None = None,
        unfreeze_n: int             = cfg.unfreeze_layers,
        pretrained: bool            = True,
        weights: list[float] | None = None,
    ):
        super().__init__()
 
        if backbone_names is None:
            backbone_names = list(self._DEFAULT_BACKBONES)
 
        # If training from scratch, every parameter should be trainable
        effective_unfreeze = -1 if not pretrained else unfreeze_n
 
        # ── Build member list ─────────────────────────────────────────────────
        members: list[nn.Module] = []
        names:   list[str]       = []
 
        for spec in backbone_names:
            if isinstance(spec, nn.Module):
                # Pre-built model passed directly — use as-is
                members.append(spec)
                names.append(type(spec).__name__)
 
            elif isinstance(spec, str) and spec == "CustomCNN":
                from src.model_components import CustomCNN
                m = CustomCNN(img_size=cfg.img_height)
                members.append(m)
                names.append("CustomCNN")
 
            elif isinstance(spec, str):
                # Any torchvision / timm backbone
                # Handle timm fallback gracefully for TinyViT
                _spec = spec
                if (not TIMM_AVAILABLE and
                        "tiny_vit" in spec.lower()):
                    print(
                        "[EnsembleModel] timm unavailable — replacing "
                        f"'{spec}' with MobileNetV3."
                    )
                    _spec = "mobilenet_v3_large"
                m = BaseModel(
                    backbone_name=_spec,
                    unfreeze_n=effective_unfreeze,
                    pretrained=pretrained,
                )
                members.append(m)
                names.append(_spec)
 
            else:
                raise TypeError(
                    f"backbone_names elements must be str or nn.Module, "
                    f"got {type(spec)}."
                )
 
        self.members_list  = nn.ModuleList(members)
        self.member_names  = names          # human-readable for diagnostics
 
        # ── Averaging weights ─────────────────────────────────────────────────
        n = len(members)
        if weights is not None:
            if len(weights) != n:
                raise ValueError(
                    f"len(weights)={len(weights)} != number of members={n}"
                )
            w = torch.tensor(weights, dtype=torch.float32)
            w = w / w.sum()
        else:
            w = torch.ones(n, dtype=torch.float32) / float(n)
        self.register_buffer("ens_weights", w)   # persisted in state_dict
 
        total_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(
            f"[EnsembleModel] Members: {self.member_names}\n"
            f"[EnsembleModel] Total trainable params: {total_trainable:,}"
        )
 
    # ── Forward ───────────────────────────────────────────────────────────────
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted-mean sigmoid probability.  Shape: (B, 1)."""
        out = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        for i, member in enumerate(self.members_list):
            logit = member(x)                             # (B, 1)
            out   = out + self.ens_weights[i] * torch.sigmoid(logit)
        return out   # probabilities in [0, 1]
 
    # ── Utilities ─────────────────────────────────────────────────────────────
 
    def member_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Raw logits from each member — for per-backbone diagnostics."""
        return [member(x) for member in self.members_list]
 
    def unfreeze_all_members(self):
        """Unfreeze every parameter in every member (call after warm-up)."""
        for member in self.members_list:
            if hasattr(member, "unfreeze_all"):
                member.unfreeze_all()          # BaseModel convenience method
            else:
                for p in member.parameters():
                    p.requires_grad = True
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EnsembleModel] All members unfrozen. Trainable: {total:,}")
 
    def members(self) -> list[nn.Module]:
        return list(self.members_list)
 
    # ── Backward-compatibility aliases ────────────────────────────────────────
 
    @property
    def eff_net(self) -> nn.Module:
        """Alias: first member (historically EfficientNet-B2)."""
        return self.members_list[0]
 
    @property
    def dense_net(self) -> nn.Module:
        """Alias: second member (historically DenseNet-121)."""
        return self.members_list[1] if len(self.members_list) > 1 else None
 
    @property
    def tiny_vit(self) -> nn.Module:
        """Alias: third member (historically TinyViT)."""
        return self.members_list[2] if len(self.members_list) > 2 else None