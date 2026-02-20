"""
Training entry point for HAT super-resolution.

Usage:
    # Single GPU smoke test
    python train.py --config configs/hat_x4.yaml --devices 1

    # Multi-GPU training (prefer scripts/train.sh)
    python train.py --config configs/hat_x4.yaml
"""

import argparse
import copy
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf

from src.data.div2k import DIV2KDataModule
from src.lightning_module import HATLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAT x4 Super-Resolution Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--devices", type=int, default=None, help="Override number of GPUs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Use W&B logger (default: TensorBoard)")
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dot notation: training.lr=1e-4 data.batch_size=4",
    )
    return parser.parse_args()


def apply_overrides(config: OmegaConf, overrides: list[str]) -> OmegaConf:
    """Apply CLI dot-notation overrides to config."""
    for override in overrides:
        key, value = override.split("=", 1)
        # Try to parse as YAML value (handles int, float, bool, etc.)
        parsed_value = OmegaConf.create({"v": value}).v
        OmegaConf.update(config, key, parsed_value)
    return config


def main() -> None:
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Seed
    seed = config_dict.get("experiment", {}).get("seed", 42)
    L.seed_everything(seed, workers=True)

    # Training config shortcuts
    train_cfg = config_dict["training"]
    devices = args.devices or train_cfg.get("devices", 1)

    # Data module
    data_cfg = config_dict["data"]
    datamodule = DIV2KDataModule(
        data_root=data_cfg["data_root"],
        scale=data_cfg.get("scale", 4),
        lr_patch_size=data_cfg.get("lr_patch_size", 64),
        batch_size=data_cfg.get("batch_size", 8),
        num_workers=data_cfg.get("num_workers", 8),
        augment=data_cfg.get("augment", True),
    )

    # Model
    model = HATLightningModule(config_dict)

    # Callbacks
    exp_name = config_dict.get("experiment", {}).get("name", "hat_x4")
    callbacks = [
        ModelCheckpoint(
            dirpath=f"experiments/{exp_name}/checkpoints",
            filename="{step:07d}-{val/psnr:.2f}",
            monitor="val/psnr",
            mode="max",
            save_top_k=train_cfg.get("save_top_k", 5),
            save_last=True,
            every_n_train_steps=train_cfg.get("val_check_interval", 5000),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # EMA via Stochastic Weight Averaging
    ema_decay = train_cfg.get("ema_decay", 0.0)
    if ema_decay > 0:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=train_cfg.get("lr", 2e-4),
                swa_epoch_start=1,
                annealing_epochs=1,
            )
        )

    # Logger
    if args.wandb:
        logger = WandbLogger(project="hat-superres", name=exp_name, save_dir="logs")
    else:
        logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    # Trainer
    strategy = train_cfg.get("strategy", "ddp") if devices > 1 else "auto"
    trainer = L.Trainer(
        max_steps=train_cfg.get("max_steps", 500000),
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=train_cfg.get("precision", "bf16-mixed"),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 4),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 100),
        val_check_interval=train_cfg.get("val_check_interval", 5000),
        callbacks=callbacks,
        logger=logger,
        benchmark=train_cfg.get("benchmark", True),
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume)

    print(f"\nTraining complete. Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
