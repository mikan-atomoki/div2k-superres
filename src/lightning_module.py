"""
Lightning Module for HAT super-resolution training.
Handles training/validation steps, metrics, and optimizer configuration.
"""

import math

import lightning as L
import torch
import torch.nn.functional as F

from src.losses.losses import CombinedLoss
from src.models.hat import HAT


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor [0,1] to YCbCr. Returns Y channel."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y = 16.0 / 255.0 + (65.481 / 255.0) * r + (128.553 / 255.0) * g + (24.966 / 255.0) * b
    return y


def calculate_psnr(sr: torch.Tensor, hr: torch.Tensor, crop_border: int = 4) -> torch.Tensor:
    """Calculate PSNR on Y channel with border cropping."""
    sr_y = rgb_to_ycbcr(sr.clamp(0, 1))
    hr_y = rgb_to_ycbcr(hr.clamp(0, 1))

    if crop_border > 0:
        sr_y = sr_y[:, :, crop_border:-crop_border, crop_border:-crop_border]
        hr_y = hr_y[:, :, crop_border:-crop_border, crop_border:-crop_border]

    mse = F.mse_loss(sr_y, hr_y)
    if mse == 0:
        return torch.tensor(100.0, device=sr.device)
    return 10.0 * torch.log10(1.0 / mse)


def calculate_ssim(sr: torch.Tensor, hr: torch.Tensor, crop_border: int = 4) -> torch.Tensor:
    """Calculate SSIM on Y channel with border cropping."""
    sr_y = rgb_to_ycbcr(sr.clamp(0, 1))
    hr_y = rgb_to_ycbcr(hr.clamp(0, 1))

    if crop_border > 0:
        sr_y = sr_y[:, :, crop_border:-crop_border, crop_border:-crop_border]
        hr_y = hr_y[:, :, crop_border:-crop_border, crop_border:-crop_border]

    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    mu_sr = F.avg_pool2d(sr_y, 11, stride=1, padding=5)
    mu_hr = F.avg_pool2d(hr_y, 11, stride=1, padding=5)

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr

    sigma_sr_sq = F.avg_pool2d(sr_y ** 2, 11, stride=1, padding=5) - mu_sr_sq
    sigma_hr_sq = F.avg_pool2d(hr_y ** 2, 11, stride=1, padding=5) - mu_hr_sq
    sigma_sr_hr = F.avg_pool2d(sr_y * hr_y, 11, stride=1, padding=5) - mu_sr_hr

    ssim_map = ((2 * mu_sr_hr + c1) * (2 * sigma_sr_hr + c2)) / \
               ((mu_sr_sq + mu_hr_sq + c1) * (sigma_sr_sq + sigma_hr_sq + c2))

    return ssim_map.mean()


class HATLightningModule(L.LightningModule):
    """
    Lightning Module wrapping HAT model with training/validation logic.
    Step-based scheduling throughout (not epoch-based).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Build model
        model_cfg = config["model"]
        self.net = HAT(
            img_size=config["data"].get("lr_patch_size", 64),
            in_chans=model_cfg.get("in_chans", 3),
            embed_dim=model_cfg.get("embed_dim", 180),
            depths=tuple(model_cfg.get("depths", [6] * 12)),
            num_heads=tuple(model_cfg.get("num_heads", [6] * 12)),
            window_size=model_cfg.get("window_size", 16),
            compress_ratio=model_cfg.get("compress_ratio", 3),
            squeeze_factor=model_cfg.get("squeeze_factor", 30),
            conv_scale=model_cfg.get("conv_scale", 0.01),
            overlap_ratio=model_cfg.get("overlap_ratio", 0.5),
            mlp_ratio=model_cfg.get("mlp_ratio", 2.0),
            upscale=model_cfg.get("upscale", 4),
            img_range=model_cfg.get("img_range", 1.0),
            upsampler=model_cfg.get("upsampler", "pixelshuffle"),
            resi_connection=model_cfg.get("resi_connection", "1conv"),
        )

        # Build loss
        loss_cfg = config.get("loss", {})
        self.criterion = CombinedLoss(
            l1_weight=loss_cfg.get("l1_weight", 1.0),
            perceptual_weight=loss_cfg.get("perceptual_weight", 0.0),
        )

        # Training config
        self.train_cfg = config.get("training", {})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        lr, hr = batch["lr"], batch["hr"]
        sr = self.net(lr)

        losses = self.criterion(sr, hr)

        self.log("train/loss", losses["total"], prog_bar=True, sync_dist=True)
        self.log("train/l1", losses["l1"], sync_dist=True)
        if "perceptual" in losses:
            self.log("train/perceptual", losses["perceptual"], sync_dist=True)

        return losses["total"]

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        lr, hr = batch["lr"], batch["hr"]

        # Pad to window_size multiple using reflect padding
        _, _, h, w = lr.shape
        window_size = self.config["model"].get("window_size", 16)
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        lr_padded = F.pad(lr, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")

        sr = self.net(lr_padded)
        # Remove padding from output
        scale = self.config["model"].get("upscale", 4)
        sr = sr[:, :, : h * scale, : w * scale]

        # Metrics on Y channel
        psnr = calculate_psnr(sr, hr, crop_border=4)
        ssim = calculate_ssim(sr, hr, crop_border=4)

        self.log("val/psnr", psnr, prog_bar=True, sync_dist=True)
        self.log("val/ssim", ssim, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> dict:
        lr = self.train_cfg.get("lr", 2e-4)
        betas = tuple(self.train_cfg.get("betas", [0.9, 0.99]))
        weight_decay = self.train_cfg.get("weight_decay", 0.0)

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )

        warmup_steps = self.train_cfg.get("warmup_steps", 5000)
        max_steps = self.train_cfg.get("max_steps", 500000)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(1e-6 / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
