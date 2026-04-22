"""CLI entry point for the triton kernel extraction pipeline.

Usage (via the bash wrapper)::

    python3 -m tools.triton_kernel_extractor \\
        --source list \\
        --dataset-base-dir /data/ai4c_dataset \\
        --graphnet-dir /opt/GraphNet \\
        --ai4c-base /opt/ai4c \\
        --graphnet-hf-dir /opt/GraphNet_hf \\
        [--gpu-ids 0 2 5 7]

When ``--gpu-ids`` is omitted the script auto-detects all available GPUs
by parsing the output of ``nvidia-smi -L``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _detect_gpu_ids() -> list[int]:
    """Auto-detect available GPU IDs.

    Priority order (matching the original bash script):
      1. ``CUDA_VISIBLE_DEVICES`` environment variable
      2. ``nvidia-smi -L`` output

    Returns a list of integer GPU indices.  Raises ``RuntimeError``
    when no GPUs are found.
    """
    # Priority 1: honour CUDA_VISIBLE_DEVICES if set.
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_env:
        try:
            return [int(x) for x in cuda_env.split(",") if x.strip()]
        except ValueError:
            pass  # Fall through to nvidia-smi.

    # Priority 2: auto-detect from nvidia-smi.
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ids = [int(m) for m in re.findall(r"GPU (\d+):", result.stdout)]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ids = []

    if not ids:
        raise RuntimeError(
            "No GPUs detected. Pass --gpu-ids explicitly or check nvidia-smi."
        )
    return ids


def _parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compile graph datasets and extract (subgraph, triton_kernel) pairs."
        ),
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=("list", "hf"),
        help="Data source type: 'list' (txt file paths) or 'hf' (scan HF dirs).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "GPU IDs to use for parallel compilation. "
            "Auto-detected via nvidia-smi when omitted."
        ),
    )
    parser.add_argument(
        "--dataset-base-dir",
        type=Path,
        required=True,
        help="Root directory of the dataset collection.",
    )
    parser.add_argument(
        "--graphnet-dir",
        type=Path,
        required=True,
        help="Path to the GraphNet repository (added to PYTHONPATH by the bash wrapper).",
    )
    parser.add_argument(
        "--ai4c-base",
        type=Path,
        required=True,
        help="Root of the ai4c repository (prefix for model paths in 'list' mode).",
    )
    parser.add_argument(
        "--graphnet-hf-dir",
        type=Path,
        required=True,
        help="Root of the GraphNet HuggingFace data directory.",
    )

    args = parser.parse_args(argv)

    gpu_ids = args.gpu_ids if args.gpu_ids else _detect_gpu_ids()

    return PipelineConfig(
        source=args.source,
        gpu_ids=gpu_ids,
        dataset_base_dir=args.dataset_base_dir,
        graphnet_dir=args.graphnet_dir,
        ai4c_base=args.ai4c_base,
        graphnet_hf_dir=args.graphnet_hf_dir,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    config = _parse_args(argv)

    logger.info("Source: %s", config.source)
    logger.info(
        "Using %d GPU(s): %s",
        len(config.gpu_ids),
        " ".join(str(g) for g in config.gpu_ids),
    )

    # Unset CUDA_VISIBLE_DEVICES in the parent process so that worker
    # subprocesses start with a clean slate and receive only the per-GPU
    # value assigned by compiler.py.  Matches the bash: `unset CUDA_VISIBLE_DEVICES`.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
