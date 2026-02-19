"""
DIV2K dataset and Lightning DataModule for paired super-resolution training.
"""

import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DIV2KDataset(Dataset):
    """
    DIV2K paired LR/HR dataset.

    Expects directory structure:
        data_root/
            DIV2K_train_HR/          # 0001.png - 0800.png
            DIV2K_train_LR_bicubic/
                X4/                  # 0001x4.png - 0800x4.png
            DIV2K_valid_HR/          # 0801.png - 0900.png
            DIV2K_valid_LR_bicubic/
                X4/                  # 0801x4.png - 0900x4.png
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        scale: int = 4,
        lr_patch_size: int = 64,
        augment: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.augment = augment and (split == "train")

        # Resolve paths
        if split == "train":
            self.hr_dir = self.data_root / "DIV2K_train_HR"
            self.lr_dir = self.data_root / "DIV2K_train_LR_bicubic" / f"X{scale}"
        else:
            self.hr_dir = self.data_root / "DIV2K_valid_HR"
            self.lr_dir = self.data_root / "DIV2K_valid_LR_bicubic" / f"X{scale}"

        # Collect file pairs
        self.hr_files = sorted(self.hr_dir.glob("*.png"))
        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No HR images found in {self.hr_dir}")

        self.lr_files = []
        for hr_f in self.hr_files:
            stem = hr_f.stem  # e.g. "0001"
            lr_f = self.lr_dir / f"{stem}x{scale}.png"
            if not lr_f.exists():
                raise FileNotFoundError(f"LR image not found: {lr_f}")
            self.lr_files.append(lr_f)

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.hr_files)

    def _paired_random_crop(self, hr: np.ndarray, lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Random crop with paired coordinates."""
        lr_h, lr_w = lr.shape[:2]
        # Random top-left for LR
        top = random.randint(0, lr_h - self.lr_patch_size)
        left = random.randint(0, lr_w - self.lr_patch_size)

        lr_crop = lr[top:top + self.lr_patch_size, left:left + self.lr_patch_size]
        # Corresponding HR crop
        hr_top, hr_left = top * self.scale, left * self.scale
        hr_crop = hr[hr_top:hr_top + self.hr_patch_size, hr_left:hr_left + self.hr_patch_size]

        return hr_crop, lr_crop

    def _augment(self, hr: np.ndarray, lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Random horizontal/vertical flip and 90-degree rotation."""
        # Horizontal flip
        if random.random() > 0.5:
            hr = np.flip(hr, axis=1)
            lr = np.flip(lr, axis=1)
        # Vertical flip
        if random.random() > 0.5:
            hr = np.flip(hr, axis=0)
            lr = np.flip(lr, axis=0)
        # 90-degree rotation
        if random.random() > 0.5:
            hr = np.transpose(hr, (1, 0, 2))
            lr = np.transpose(lr, (1, 0, 2))
        return hr, lr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load images
        hr = np.array(Image.open(self.hr_files[idx]).convert("RGB"))
        lr = np.array(Image.open(self.lr_files[idx]).convert("RGB"))

        if self.split == "train":
            hr, lr = self._paired_random_crop(hr, lr)
            if self.augment:
                hr, lr = self._augment(hr, lr)

        # Ensure contiguous arrays for ToTensor
        hr = np.ascontiguousarray(hr)
        lr = np.ascontiguousarray(lr)

        hr_tensor = self.to_tensor(hr)  # [0, 1] float32
        lr_tensor = self.to_tensor(lr)

        return {"lr": lr_tensor, "hr": hr_tensor}


class DIV2KDataModule(L.LightningDataModule):
    """Lightning DataModule for DIV2K dataset with DDP support."""

    def __init__(
        self,
        data_root: str = "data/DIV2K",
        scale: int = 4,
        lr_patch_size: int = 64,
        batch_size: int = 8,
        num_workers: int = 8,
        augment: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_root = data_root
        self.scale = scale
        self.lr_patch_size = lr_patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = DIV2KDataset(
                data_root=self.data_root,
                split="train",
                scale=self.scale,
                lr_patch_size=self.lr_patch_size,
                augment=self.augment,
            )
        if stage in ("fit", "validate", None):
            self.val_dataset = DIV2KDataset(
                data_root=self.data_root,
                split="valid",
                scale=self.scale,
                lr_patch_size=self.lr_patch_size,
                augment=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
