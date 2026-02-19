#!/usr/bin/env bash
set -euo pipefail

# Download DIV2K dataset for x4 super-resolution training.
# Source: ETH Zurich official servers

DATA_DIR="${1:-data/DIV2K}"
SCALE=4

BASE_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

echo "=== Downloading DIV2K Dataset ==="
echo "Target directory: $(pwd)"

# HR training images
if [ ! -d "DIV2K_train_HR" ]; then
    echo "Downloading training HR images..."
    wget -c "${BASE_URL}/DIV2K_train_HR.zip"
    unzip -q DIV2K_train_HR.zip
    rm DIV2K_train_HR.zip
else
    echo "Training HR images already exist, skipping."
fi

# LR training images (bicubic x4)
if [ ! -d "DIV2K_train_LR_bicubic/X${SCALE}" ]; then
    echo "Downloading training LR images (bicubic x${SCALE})..."
    wget -c "${BASE_URL}/DIV2K_train_LR_bicubic_X${SCALE}.zip"
    unzip -q "DIV2K_train_LR_bicubic_X${SCALE}.zip"
    rm "DIV2K_train_LR_bicubic_X${SCALE}.zip"
else
    echo "Training LR images already exist, skipping."
fi

# HR validation images
if [ ! -d "DIV2K_valid_HR" ]; then
    echo "Downloading validation HR images..."
    wget -c "${BASE_URL}/DIV2K_valid_HR.zip"
    unzip -q DIV2K_valid_HR.zip
    rm DIV2K_valid_HR.zip
else
    echo "Validation HR images already exist, skipping."
fi

# LR validation images (bicubic x4)
if [ ! -d "DIV2K_valid_LR_bicubic/X${SCALE}" ]; then
    echo "Downloading validation LR images (bicubic x${SCALE})..."
    wget -c "${BASE_URL}/DIV2K_valid_LR_bicubic_X${SCALE}.zip"
    unzip -q "DIV2K_valid_LR_bicubic_X${SCALE}.zip"
    rm "DIV2K_valid_LR_bicubic_X${SCALE}.zip"
else
    echo "Validation LR images already exist, skipping."
fi

echo ""
echo "=== Download Complete ==="
echo "Directory structure:"
echo "  ${DATA_DIR}/DIV2K_train_HR/         (800 images)"
echo "  ${DATA_DIR}/DIV2K_train_LR_bicubic/X${SCALE}/  (800 images)"
echo "  ${DATA_DIR}/DIV2K_valid_HR/         (100 images)"
echo "  ${DATA_DIR}/DIV2K_valid_LR_bicubic/X${SCALE}/  (100 images)"
