"""
Inference script for HAT super-resolution.

Supports:
  - Single image or directory batch processing
  - Tile-based inference for large images (avoids OOM)
  - Automatic padding for window_size alignment

Usage:
    python inference.py --checkpoint experiments/hat_l_x4_div2k/checkpoints/last.ckpt \
                        --input test_images/ --output results/ --tile_size 256
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.lightning_module import HATLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAT Super-Resolution Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--tile_size", type=int, default=0,
                        help="Tile size for large images (0 = no tiling)")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Tile overlap in pixels")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> HATLightningModule:
    """Load model from Lightning checkpoint."""
    model = HATLightningModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def pad_to_window(img: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    """Pad image to be divisible by window_size. Returns padded image and original h, w."""
    _, _, h, w = img.shape
    mod_h = (window_size - h % window_size) % window_size
    mod_w = (window_size - w % window_size) % window_size
    img = F.pad(img, (0, mod_w, 0, mod_h), mode="reflect")
    return img, h, w


@torch.no_grad()
def inference_single(model: HATLightningModule, img: torch.Tensor) -> torch.Tensor:
    """Run inference on a single image (no tiling)."""
    window_size = model.config["model"].get("window_size", 16)
    scale = model.config["model"].get("upscale", 4)

    img, orig_h, orig_w = pad_to_window(img, window_size)
    sr = model(img)
    sr = sr[:, :, : orig_h * scale, : orig_w * scale]
    return sr.clamp(0, 1)


@torch.no_grad()
def inference_tiled(
    model: HATLightningModule, img: torch.Tensor,
    tile_size: int, tile_overlap: int,
) -> torch.Tensor:
    """Tile-based inference for large images."""
    scale = model.config["model"].get("upscale", 4)
    window_size = model.config["model"].get("window_size", 16)
    _, _, h, w = img.shape

    # Ensure tile_size is divisible by window_size
    tile_size = (tile_size // window_size) * window_size
    stride = tile_size - tile_overlap

    out_h, out_w = h * scale, w * scale
    output = torch.zeros(1, 3, out_h, out_w, device=img.device)
    weight = torch.zeros(1, 1, out_h, out_w, device=img.device)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(y_end - tile_size, 0)
            x_start = max(x_end - tile_size, 0)

            tile = img[:, :, y_start:y_end, x_start:x_end]

            # Pad tile if needed
            tile, tile_h, tile_w = pad_to_window(tile, window_size)

            # Inference
            sr_tile = model(tile)
            sr_tile = sr_tile[:, :, : tile_h * scale, : tile_w * scale]
            sr_tile = sr_tile.clamp(0, 1)

            # Place tile in output
            out_y = y_start * scale
            out_x = x_start * scale
            out_tile_h = (y_end - y_start) * scale
            out_tile_w = (x_end - x_start) * scale

            output[:, :, out_y:out_y + out_tile_h, out_x:out_x + out_tile_w] += sr_tile
            weight[:, :, out_y:out_y + out_tile_h, out_x:out_x + out_tile_w] += 1

    output = output / weight
    return output


def process_image(
    model: HATLightningModule, input_path: Path, output_path: Path,
    tile_size: int, tile_overlap: int, device: str,
) -> None:
    """Process a single image."""
    to_tensor = transforms.ToTensor()
    img = Image.open(input_path).convert("RGB")
    img_tensor = to_tensor(img).unsqueeze(0).to(device)

    start = time.time()

    if tile_size > 0:
        sr = inference_tiled(model, img_tensor, tile_size, tile_overlap)
    else:
        sr = inference_single(model, img_tensor)

    elapsed = time.time() - start

    # Save
    sr_img = transforms.ToPILImage()(sr.squeeze(0).cpu())
    sr_img.save(output_path, quality=95)

    scale = model.config["model"].get("upscale", 4)
    h, w = img.size[1], img.size[0]
    print(f"  {input_path.name}: {w}x{h} -> {w*scale}x{h*scale} ({elapsed:.2f}s)")


def main() -> None:
    args = parse_args()

    print("Loading model...")
    model = load_model(args.checkpoint, args.device)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input images
    if input_path.is_file():
        images = [input_path]
    else:
        images = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        )

    if not images:
        print(f"No images found in {input_path}")
        return

    print(f"Processing {len(images)} image(s)...")
    tile_info = f" (tile_size={args.tile_size})" if args.tile_size > 0 else ""
    print(f"Device: {args.device}{tile_info}")

    for img_path in images:
        out_path = output_dir / f"{img_path.stem}_sr.png"
        process_image(model, img_path, out_path, args.tile_size, args.tile_overlap, args.device)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
