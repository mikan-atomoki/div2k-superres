#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-hat-sr}"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.1"

echo "=== HAT x4 Super-Resolution Setup ==="
echo "Environment: ${ENV_NAME}"

# Create conda environment
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

# Install remaining dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Create directories
echo "Creating data and output directories..."
mkdir -p data/DIV2K experiments logs

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Download DIV2K: bash scripts/download_div2k.sh"
echo "Start training: bash scripts/train.sh"
