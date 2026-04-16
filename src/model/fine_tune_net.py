"""Fine-tuning helpers for pretrained torchvision models.

Supported backbones: alexnet, resnet18, resnet50, vgg16, efficientnet_b0.
The last classification layer is replaced to output a single DR score in [0, 1].
"""

import torch.nn as nn
from torchvision import models


def build_fine_tune_model(backbone: str = 'alexnet') -> nn.Module:
    """Load a pretrained torchvision model and adapt it for binary DR classification.

    Args:
        backbone: one of 'alexnet', 'resnet18', 'resnet50', 'vgg16', 'efficientnet_b0'.

    Returns:
        nn.Module with the final layer replaced by Linear(..., 1) + Sigmoid.
    """
    backbone = backbone.lower()

    if backbone == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    elif backbone == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    elif backbone == 'resnet18':
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            "Choose from: alexnet, resnet18, resnet50, vgg16, efficientnet_b0."
        )

    return model
