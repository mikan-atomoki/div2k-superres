#!/usr/bin/env bash
set -euo pipefail

# Launch distributed HAT training on multiple GPUs.
# Usage:
#   bash scripts/train.sh                          # defaults: 8 GPUs, hat_x4.yaml
#   bash scripts/train.sh --devices 4              # override GPU count
#   bash scripts/train.sh --config my_config.yaml  # custom config
#   bash scripts/train.sh --resume path/to/ckpt    # resume training

CONFIG="${CONFIG:-configs/hat_x4.yaml}"
EXTRA_ARGS=("$@")

echo "=== HAT x4 Training ==="
echo "Config: ${CONFIG}"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"

python train.py --config "${CONFIG}" "${EXTRA_ARGS[@]}"
