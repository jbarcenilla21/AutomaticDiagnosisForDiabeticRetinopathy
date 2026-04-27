# =============================================================================
# model.py  —  BaseModel (single fine-tuned backbone) + EnsembleModel
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
from src.model_components import CustomVGG, SimpleLeNet
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print(
        "[model.py] WARNING: 'timm' not installed.  "
        "TinyViT will be replaced by MobileNetV3.\n"
        "  Install with:  pip install timm"
    )

from utils.config import Config as cfg


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
        unfreeze_n:    int  = cfg.unfreeze_layers,
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
            # timm models already set num_classes at creation
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
            return   # train everything from scratch

        # Freeze all parameters first
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze the last unfreeze_n layer groups
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

        Member 0  →  interpolated to ensemble_scales[0]  (224 — global context)
        Member 1  →  interpolated to ensemble_scales[1]  (384 — mid-scale)
        Member 2  →  interpolated to ensemble_scales[2]  (512 — fine lesions)

    Each member outputs a raw logit.  sigmoid converts it to a probability.
    The ensemble output is the LEARNABLE-WEIGHTED mean probability across members.

    Ensemble weights (``ens_weights``) are ``nn.Parameter`` values; Softmax is
    applied in forward() so they always sum to 1 and stay positive.  This
    allows the network to learn which backbone / scale is most reliable.

    Args:
        backbone_names: List of backbone specs (str / ``"Custom_VGG"`` / nn.Module).
                        Defaults to EfficientNet-B2 + DenseNet-121 + TinyViT.
        unfreeze_n:     Layer groups to unfreeze per string-based member.
                        Forced to -1 when pretrained=False.
        pretrained:     Load ImageNet weights for string-based members.
        weights:        Optional initial weights [w0, w1, ...].  Uniform if None.
                        These are the raw (pre-softmax) values of ens_weights.
        pth_path:       Optional path to load a checkpoint state_dict for the ensemble.
    """

    _DEFAULT_BACKBONES = [
        "efficientnet_b2",
        "densenet121",
        "tiny_vit_21m_224.dist_in22k_ft_in1k",
    ]

    def __init__(
        self,
        backbone_names: list | None         = None,
        unfreeze_n:     int                 = cfg.unfreeze_layers,
        pretrained:     bool                = True,
        weights:        list[float] | None  = None,
        pth_path:       Path | None         = None,
    ):
        super().__init__()

        # Load arquitecture
        self._load_from_scratch(backbone_names, pretrained, unfreeze_n, weights)
        
        # Load checkpoint if provided
        if pth_path is not None and os.path.exists(pth_path):
            print(f"[EnsembleModel] Loading checkpoint from {pth_path}...")
            state_dict = torch.load(pth_path, map_location="cpu")
            self.load_state_dict(state_dict)

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[EnsembleModel] Members   : {self.member_names}\n"
            f"[EnsembleModel] Scales    : {cfg.ensemble_scales}\n"
            f"[EnsembleModel] Trainable : {total_trainable:,}"
        )
    # ── load functioanlity ────────────────────────────────────────────────────

    def _load_from_scratch(self, backbone_names, pretrained, unfreeze_n, weights):
        if backbone_names is None:
            backbone_names = list(self._DEFAULT_BACKBONES)

        # When training from scratch every parameter must be trainable
        effective_unfreeze = -1 if not pretrained else unfreeze_n

        # ── Build member list ─────────────────────────────────────────────────
        members: list[nn.Module] = []
        names:   list[str]       = []

        for spec in backbone_names:
            if isinstance(spec, nn.Module):
                members.append(spec)
                names.append(type(spec).__name__)

            elif isinstance(spec, str) and spec.lower() == "customvgg":
                m = CustomVGG(img_size=cfg.img_height)
                members.append(m)
                names.append("customVGG")

            elif isinstance(spec, str) and spec.lower() == "simplelenet":
                m = SimpleLeNet(img_size=cfg.img_height)
                members.append(m)
                names.append("simpleLeNet")

            elif isinstance(spec, str) and pretrained == True:
                # Handle missing timm for TinyViT gracefully
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
        self.member_names = names      # human-readable labels for diagnostics

        # ── Learnable ensemble weights (nn.Parameter, Softmax in forward) ─────
        n = len(members)
        if weights is not None:
            if len(weights) != n:
                raise ValueError(f"len(weights)={len(weights)} != num members={n}")
            w = torch.tensor(weights, dtype=torch.float32)
        else:
            # Start with uniform weights (all zeros → softmax gives 1/n each)
            w = torch.zeros(n, dtype=torch.float32)
        # nn.Parameter → included in model.parameters() and optimiser updates
        self.ens_weights = nn.Parameter(w)
    

    def _load_from_checkpoint(self, pth_path: Path):
        checkpoint = torch.load(pth_path, map_location=cfg.device)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format")

        # limpiar DataParallel
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        print("[EnsembleModel] Checkpoint loaded")
        print(f"  Missing keys   : {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
    
    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-scale weighted-mean probability.  Shape: (B, 1).

        Each member receives the input resized to its designated resolution
        (from cfg.ensemble_scales).  Learnable Softmax weights control the
        contribution of each member to the final probability.
        """
        # Normalise learnable weights with Softmax so they sum to 1
        w = torch.softmax(self.ens_weights, dim=0)   # shape (n_members,)

        out = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

        for i, member in enumerate(self.members_list):
            # Select the scale assigned to this member (cycling if needed)
            target_res = cfg.ensemble_scales[i % len(cfg.ensemble_scales)]

            # Resize input to this member's expected resolution
            x_res = F.interpolate(
                x, size=(target_res, target_res),
                mode="bilinear", align_corners=False,
            )

            logit = member(x_res)                          # (B, 1) raw logit
            prob  = torch.sigmoid(logit)                   # (B, 1) probability
            out   = out + w[i] * prob

        return out   # weighted-mean probability in [0, 1]

    # ── Test-Time Augmentation ─────────────────────────────────────────────────

    @torch.no_grad()
    def predict_tta(
        self,
        x:        torch.Tensor,
        n_passes: int = cfg.num_tta,
    ) -> torch.Tensor:
        """Run N augmented forward passes and return the mean probability.

        Augmentations applied in tensor space (no PIL conversion):
            • Random horizontal flip (50%)
            • Random vertical flip   (50%)
            • Small random 90° rotation (25% each direction)

        Args:
            x:        Input batch (B, 3, H, W) already on the correct device.
            n_passes: Number of TTA passes.  1 → deterministic single pass.

        Returns:
            Mean probability tensor (B, 1).
        """
        self.eval()
        prob_sum = torch.zeros(x.size(0), 1, device=x.device)

        for _ in range(n_passes):
            x_aug = x.clone()

            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                x_aug = torch.flip(x_aug, dims=[3])

            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                x_aug = torch.flip(x_aug, dims=[2])

            # Random 90° rotation (keeps retinal structures realistic)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x_aug = torch.rot90(x_aug, k=k, dims=[2, 3])

            prob_sum = prob_sum + self.forward(x_aug)

        return prob_sum / n_passes   # (B, 1) mean probability

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def member_logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Raw logits from each member — for per-backbone AUC diagnostics.

        NOTE: Each member receives the input at its designated scale,
        matching the behaviour of forward().
        """
        logits = []
        for i, member in enumerate(self.members_list):
            target_res = cfg.ensemble_scales[i % len(cfg.ensemble_scales)]
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
        """First member (historically EfficientNet-B2)."""
        return self.members_list[0]

    @property
    def dense_net(self) -> nn.Module:
        """Second member (historically DenseNet-121)."""
        return self.members_list[1] if len(self.members_list) > 1 else None

    @property
    def tiny_vit(self) -> nn.Module:
        """Third member (historically TinyViT)."""
        return self.members_list[2] if len(self.members_list) > 2 else None