#!/bin/bash
set -euo pipefail

# Thin launcher for the triton kernel extraction pipeline.
#
# This script sets machine-specific paths and delegates all logic to the
# Python module at tools/triton_kernel_extractor.
#
# Usage:
#   bash extract_triton_kernels.sh <source> [gpu_ids]
#
# Args:
#   source   (required): "list" or "hf"
#   gpu_ids  (optional): comma-separated GPU IDs, e.g. "0,2,5,7"
#
# Examples:
#   bash extract_triton_kernels.sh list            # list source, auto-detect GPUs
#   bash extract_triton_kernels.sh hf 0,2,5,7      # hf source, specified GPUs

# ============================================================
# Arguments
# ============================================================

SOURCE="${1:?Usage: bash extract_triton_kernels.sh <source> [gpu_ids]  (source: list | hf)}"
GPU_ARG="${2:-}"

# ============================================================
# Machine-specific path configuration
#
# Edit the variables below to match your local environment.
# ============================================================

DATASET_BASE_DIR="/path/to/dataset_output"
GRAPHNET_DIR="/path/to/GraphNet/GitHub/repo/"

# ============================================================
# Environment setup
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASSNET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$GRAPHNET_DIR:${PASSNET_DIR}:${PYTHONPATH:-}"

# ============================================================
# Build Python CLI arguments
# ============================================================

PYTHON_ARGS=(
    --source "$SOURCE"
    --dataset-base-dir "$DATASET_BASE_DIR"
    --graphnet-dir "$GRAPHNET_DIR"
    --passnet-dir "$PASSNET_DIR"
    --max-autotune
    --enable-cache-analysis
)

if [ -n "$GPU_ARG" ]; then
    # Convert comma-separated "0,2,5,7" to space-separated args.
    IFS=',' read -ra GPU_IDS <<< "$GPU_ARG"
    PYTHON_ARGS+=(--gpu-ids "${GPU_IDS[@]}")
fi

# ============================================================
# Run
# ============================================================

exec python3 -m tools.triton_kernel_extractor "${PYTHON_ARGS[@]}"
