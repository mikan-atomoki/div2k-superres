"""
Loss functions for super-resolution training.
L1Loss + optional Perceptual Loss (VGG19-based).
"""

import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG19 features.

    Extracts features from layers: relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
    and computes L1 distance between SR and HR features.
    """

    LAYER_WEIGHTS = {
        "3": 0.1,   # relu1_2
        "8": 0.1,   # relu2_2
        "17": 1.0,  # relu3_4
        "26": 1.0,  # relu4_4
        "35": 1.0,  # relu5_4
    }

    def __init__(self, layer_weights: dict[str, float] | None = None):
        super().__init__()
        self.layer_weights = layer_weights or self.LAYER_WEIGHTS

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # Freeze
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.vgg.eval()

        self.max_layer = max(int(k) for k in self.layer_weights) + 1
        self.criterion = nn.L1Loss()

        # VGG normalization constants
        self.register_buffer(
            "vgg_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "vgg_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [0, 1] to VGG input range."""
        return (x - self.vgg_mean) / self.vgg_std

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr = self._normalize(sr)
        hr = self._normalize(hr)

        loss = torch.tensor(0.0, device=sr.device)
        sr_feat = sr
        hr_feat = hr

        for i in range(self.max_layer):
            sr_feat = self.vgg[i](sr_feat)
            hr_feat = self.vgg[i](hr_feat)

            if str(i) in self.layer_weights:
                loss = loss + self.layer_weights[str(i)] * self.criterion(sr_feat, hr_feat.detach())

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of L1 and optional Perceptual loss.

    Args:
        l1_weight: Weight for L1 loss (default: 1.0).
        perceptual_weight: Weight for perceptual loss (default: 0.0, disabled).
    """

    def __init__(self, l1_weight: float = 1.0, perceptual_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

        self.l1_loss = nn.L1Loss()

        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = {}

        l1 = self.l1_loss(sr, hr)
        losses["l1"] = l1
        total = self.l1_weight * l1

        if self.perceptual_loss is not None:
            percep = self.perceptual_loss(sr, hr)
            losses["perceptual"] = percep
            total = total + self.perceptual_weight * percep

        losses["total"] = total
        return losses
