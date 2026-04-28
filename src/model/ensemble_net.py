# =============================================================================
# ensemble_net.py  —  BaseModel (single fine-tuned backbone) + EnsembleModel
# =============================================================================
"""
Architecture overview
─────────────────────
BaseModel
    Wraps any torchvision / timm model.  The last ``unfreeze_n`` layer groups
    are unfrozen; the rest stay frozen.  The classification head is replaced
    with a single linear output (raw logit for BCEWithLogitsLoss / FocalLoss).

    Supported torchvision backbones:
        efficientnet_b2, efficientnet_b3, densenet121,
        mobilenet_v3_large, vgg16, vgg16_bn, resnet50, resnext50_32x4d

    Any timm model is also supported when timm is installed.

EnsembleModel
    Heterogeneous soft-voting ensemble.  Accepts a list of backbone spec strings,
    the special string "Custom_VGG", or pre-built nn.Module instances.

    Multi-scale forward pass:
        Model 0 → interpolated to ensemble_scales[0] = 224 (global context)
        Model 1 → interpolated to ensemble_scales[1] = 384 (mid-scale)
        Model 2 → interpolated to ensemble_scales[2] = 512 (fine lesions)

    Ensemble weights are LEARNABLE (nn.Parameter with Softmax normalisation),
    so the model learns which scale/backbone is most reliable.

    predict_tta(): performs N augmented forward passes (random flips/rotations)
    and returns the mean probability — used at test time.
"""

from __future__ import annotations
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.model.custom_net import CustomVGG, SimpleLeNet

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print(
        "[ensemble_net.py] WARNING: 'timm' not installed.  "
        "TinyViT will be replaced by MobileNetV3.\n"
        "  Install with:  pip install timm"
    )

# Default ensemble scales (one per member); imported by EnsembleModel
_DEFAULT_ENSEMBLE_SCALES = [224, 384, 512]
_DEFAULT_NUM_TTA = 10
_DEFAULT_UNFREEZE_LAYERS = 2


# =============================================================================
# Utility helpers
# =============================================================================

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
        in_feat = model.head.in_features
        model.head = nn.Linear(in_feat, out_features)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'.  "
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
        return list(model.stages) + [model.head]
    elif "vit" in name or "swin" in name:
        return list(model.blocks) + [model.head]
    else:
        return list(model.children())


# =============================================================================
# BaseModel  —  single fine-tuned backbone
# =============================================================================

class BaseModel(nn.Module):
    """Single pretrained backbone fine-tuned for binary DR classification.

    Args:
        backbone_name: e.g. ``"efficientnet_b2"``, ``"densenet121"``,
                       or any timm model name.
        unfreeze_n:    Number of layer groups (from the end) to leave trainable.
                       -1 → unfreeze the entire network (Custom track).
        pretrained:    Load ImageNet weights.  False → random init.
    """

    def __init__(
        self,
        backbone_name: str  = "efficientnet_b2",
        unfreeze_n:    int  = _DEFAULT_UNFREEZE_LAYERS,
        pretrained:    bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.model         = self._build_backbone(backbone_name, pretrained)
        self._freeze_and_unfreeze(unfreeze_n)
        print(f"[BaseModel] {backbone_name} | {_count_params(self.model)}")

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
            return m
        else:
            raise ValueError(
                f"Backbone '{name}' requires timm.  "
                "Install with:  pip install timm"
            )

        _replace_head(m, name_lower)
        return m

    def _freeze_and_unfreeze(self, unfreeze_n: int):
        if unfreeze_n == -1:
            return

        for p in self.model.parameters():
            p.requires_grad = False

        groups = _get_layer_groups(self.model, self.backbone_name)
        print(f"[BaseModel] Total groups: {len(groups)}.  Unfreezing last {unfreeze_n}.")
        for group in groups[-unfreeze_n:]:
            for p in group.parameters():
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # raw logit → (B, 1)

    def unfreeze_all(self):
        """Unfreeze the full network (call after initial warm-up phase)."""
        for p in self.model.parameters():
            p.requires_grad = True
        print(f"[BaseModel] All layers unfrozen. {_count_params(self.model)}")


# =============================================================================
# EnsembleModel  —  Heterogeneous multi-scale soft-voting ensemble
# =============================================================================

class EnsembleModel(nn.Module):
    """Multi-scale heterogeneous soft-voting ensemble.

    Each member receives the input at a DIFFERENT resolution, encouraging
    complementary feature extraction:

        Member 0  →  224px  (global context)
        Member 1  →  384px  (mid-scale)
        Member 2  →  512px  (fine lesions)

    Ensemble weights (``ens_weights``) are ``nn.Parameter`` values; Softmax is
    applied in forward() so they always sum to 1 and stay positive.

    Args:
        backbone_names: List of backbone specs (str / ``"CustomVGG"`` / nn.Module).
                        Defaults to EfficientNet-B2 + DenseNet-121 + TinyViT.
        unfreeze_n:     Layer groups to unfreeze per string-based member.
        pretrained:     Load ImageNet weights for string-based members.
        weights:        Optional initial weights [w0, w1, ...].  Uniform if None.
        pth_path:       Optional path to load a checkpoint state_dict.
        ensemble_scales: Resolution per member. Defaults to [224, 384, 512].
    """

    _DEFAULT_BACKBONES = [
        "efficientnet_b2",
        "densenet121",
        "tiny_vit_21m_224.dist_in22k_ft_in1k",
    ]

    def __init__(
        self,
        backbone_names:  list | None         = None,
        unfreeze_n:      int                 = _DEFAULT_UNFREEZE_LAYERS,
        pretrained:      bool                = True,
        weights:         list[float] | None  = None,
        pth_path:        Path | None         = None,
        ensemble_scales: list[int] | None    = None,
        num_tta:         int                 = _DEFAULT_NUM_TTA,
    ):
        super().__init__()
        self.ensemble_scales = ensemble_scales or list(_DEFAULT_ENSEMBLE_SCALES)
        self.num_tta         = num_tta

        self._load_from_scratch(backbone_names, pretrained, unfreeze_n, weights)

        if pth_path is not None and os.path.exists(pth_path):
            print(f"[EnsembleModel] Loading checkpoint from {pth_path}...")
            state_dict = torch.load(pth_path, map_location="cpu")
            self.load_state_dict(state_dict)

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[EnsembleModel] Members   : {self.member_names}\n"
            f"[EnsembleModel] Scales    : {self.ensemble_scales}\n"
            f"[EnsembleModel] Trainable : {total_trainable:,}"
        )

    def _load_from_scratch(self, backbone_names, pretrained, unfreeze_n, weights):
        if backbone_names is None:
            backbone_names = list(self._DEFAULT_BACKBONES)

        effective_unfreeze = -1 if not pretrained else unfreeze_n

        members: list[nn.Module] = []
        names:   list[str]       = []

        for spec in backbone_names:
            if isinstance(spec, nn.Module):
                members.append(spec)
                names.append(type(spec).__name__)

            elif isinstance(spec, str) and spec.lower() == "customvgg":
                members.append(CustomVGG())
                names.append("customVGG")

            elif isinstance(spec, str) and spec.lower() == "simplelenet":
                members.append(SimpleLeNet())
                names.append("simpleLeNet")

            elif isinstance(spec, str) and pretrained:
                _spec = spec
                if not TIMM_AVAILABLE and "tiny_vit" in spec.lower():
                    print(f"[EnsembleModel] timm unavailable — replacing '{spec}' with MobileNetV3.")
                    _spec = "mobilenet_v3_large"
                m = BaseModel(
                    backbone_name = _spec,
                    unfreeze_n    = effective_unfreeze,
                    pretrained    = pretrained,
                )
                members.append(m)
                names.append(_spec)

            else:
                raise TypeError(
                    f"backbone_names elements must be str or nn.Module, got {type(spec)}."
                )

        self.members_list = nn.ModuleList(members)
        self.member_names = names

        n = len(members)
        if weights is not None:
            if len(weights) != n:
                raise ValueError(f"len(weights)={len(weights)} != num members={n}")
            w = torch.tensor(weights, dtype=torch.float32)
        else:
            w = torch.zeros(n, dtype=torch.float32)
        self.ens_weights = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale weighted-mean probability.  Shape: (B, 1)."""
        w   = torch.softmax(self.ens_weights, dim=0)
        out = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

        for i, member in enumerate(self.members_list):
            target_res = self.ensemble_scales[i % len(self.ensemble_scales)]
            x_res = F.interpolate(
                x, size=(target_res, target_res),
                mode="bilinear", align_corners=False,
            )
            logit = member(x_res)
            prob  = torch.sigmoid(logit)
            out   = out + w[i] * prob

        return out   # weighted-mean probability in [0, 1]

    @torch.no_grad()
    def predict_tta(self, x: torch.Tensor, n_passes: int | None = None) -> torch.Tensor:
        """Run N augmented forward passes and return the mean probability."""
        n_passes = n_passes or self.num_tta
        self.eval()
        prob_sum = torch.zeros(x.size(0), 1, device=x.device)

        for _ in range(n_passes):
            x_aug = x.clone()
            if torch.rand(1).item() > 0.5:
                x_aug = torch.flip(x_aug, dims=[3])
            if torch.rand(1).item() > 0.5:
                x_aug = torch.flip(x_aug, dims=[2])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x_aug = torch.rot90(x_aug, k=k, dims=[2, 3])
            prob_sum = prob_sum + self.forward(x_aug)

        return prob_sum / n_passes

    def member_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Raw logits from each member — for per-backbone AUC diagnostics."""
        logits = []
        for i, member in enumerate(self.members_list):
            target_res = self.ensemble_scales[i % len(self.ensemble_scales)]
            x_res      = F.interpolate(
                x, size=(target_res, target_res),
                mode="bilinear", align_corners=False,
            )
            logits.append(member(x_res))
        return logits

    def unfreeze_all_members(self):
        """Unfreeze every parameter in every member (call after warm-up)."""
        for member in self.members_list:
            if hasattr(member, "unfreeze_all"):
                member.unfreeze_all()
            else:
                for p in member.parameters():
                    p.requires_grad = True
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EnsembleModel] All members unfrozen.  Trainable: {total:,}")

    def members(self) -> list[nn.Module]:
        return list(self.members_list)

    # ── Backward-compatible property aliases ──────────────────────────────────

    @property
    def eff_net(self) -> nn.Module:
        return self.members_list[0]

    @property
    def dense_net(self) -> nn.Module:
        return self.members_list[1] if len(self.members_list) > 1 else None

    @property
    def tiny_vit(self) -> nn.Module:
        return self.members_list[2] if len(self.members_list) > 2 else None
