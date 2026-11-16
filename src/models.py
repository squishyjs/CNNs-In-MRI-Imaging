"""The Five model configurations A–E (see Table 3.3, implementation guide)"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class SliceMLP(nn.Module):
    """Model A: fully connected baseline on 64×64 slices, max-pooled over slices."""

    def __init__(self, n_out: int = 3, p: float = 0.5):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 64, 256), nn.ReLU(),
                                 nn.Dropout(p), nn.Linear(256, 128), nn.ReLU())
        self.head = nn.Linear(128, n_out)

    def forward(self, x):                      # x: (S, 1, 64, 64), one exam
        return self.head(self.enc(x).max(dim=0).values)


class SliceMaxNet(nn.Module):
    """Models B, C and E: ImageNet-pretrained slice encoder, global average pooling,
    element-wise max over slices, linear head (MRNet-style aggregation, §3.5)."""

    def __init__(self, backbone: str = "resnet18", n_out: int = 3, p: float = 0.0):
        super().__init__()
        if backbone == "alexnet":
            self.enc, dim = tvm.alexnet(weights="IMAGENET1K_V1").features, 256
        else:
            net = tvm.resnet18(weights="IMAGENET1K_V1")
            self.enc, dim = nn.Sequential(*list(net.children())[:-2]), 512
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(dim, n_out))

    def forward(self, x):                      # x: (S, 3, 224, 224), one exam
        f = self.pool(self.enc(x)).flatten(1)  # (S, dim)
        return self.head(f.max(dim=0).values)


class VolDenseNet(nn.Module):
    """Model D: MONAI volumetric DenseNet-121, input (B, 1, 32, 128, 128)."""

    def __init__(self, n_out: int = 3, p: float = 0.0):
        super().__init__()
        from monai.networks.nets import DenseNet121
        self.net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_out,
                               dropout_prob=p)

    def forward(self, x):
        return self.net(x)


def build_model(model_id: str, n_out: int, dropout: float | None = None) -> nn.Module:
    if model_id == "A":
        return SliceMLP(n_out=n_out, p=0.5 if dropout is None else dropout)
    dropout = dropout or 0.0
    if model_id == "B":
        return SliceMaxNet("alexnet", n_out=n_out, p=dropout)
    if model_id in ("C", "E"):
        return SliceMaxNet("resnet18", n_out=n_out, p=dropout)
    if model_id == "D":
        return VolDenseNet(n_out=n_out, p=dropout)
    raise ValueError(model_id)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gradcam_target(model: nn.Module, model_id: str) -> nn.Module:
    """Module whose output feature maps Grad-CAM hooks (last conv stage)."""
    if model_id == "D":
        return model.net.features
    return model.enc