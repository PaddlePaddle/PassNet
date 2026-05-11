"""Enumerate graph samples from 'list' or 'hf' data sources."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def enumerate_list_samples(graph_list_file: Path) -> list[str]:
    """Read sample paths from a text file, one per line.

    Blank lines are silently skipped.

    Raises:
        FileNotFoundError: If *graph_list_file* does not exist.
    """
    lines: list[str] = []
    with open(graph_list_file, encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if stripped:
                lines.append(stripped)
    return lines


def enumerate_hf_samples(passnet_hf_dir: Path, hf_subdir: str) -> list[str]:
    """Discover samples by scanning for ``model.py`` under a HuggingFace dir.

    Returns the sorted list of parent directories that contain a ``model.py``.

    Raises:
        FileNotFoundError: If the base directory does not exist.
    """
    base_dir = passnet_hf_dir / hf_subdir
    if not base_dir.is_dir():
        raise FileNotFoundError(f"HF dataset directory not found: {base_dir}")
    parents = sorted(
        {str(p.parent) for p in base_dir.rglob("model.py") if p.is_file()}
    )
    return parents


def compute_unique_dir(
    source: str,
    sample_path: str,
    passnet_hf_dir: str,
) -> str:
    """Derive a flat directory name from a sample path.

    For *list* sources the entire ``sample_path`` has ``/`` replaced by ``_``.
    For *hf* sources only the relative portion below ``passnet_hf_dir`` is used.

    This mirrors the bash logic::

        list:  unique_dir="${sample_path//\\//_}"
        hf:    rel_path="${sample_path#$PASSNET_HF_DIR/}"
               unique_dir="${rel_path//\\//_}"
    """
    if source == "list":
        return sample_path.replace("/", "_")

    # source == "hf"
    hf_prefix = passnet_hf_dir.rstrip("/") + "/"
    if sample_path.startswith(hf_prefix):
        rel = sample_path[len(hf_prefix):]
    else:
        rel = sample_path
    return rel.replace("/", "_")


def resolve_model_path(
    source: str,
    sample_path: str,
    passnet_dir: str,
) -> str:
    """Return the absolute path to the model directory.

    For *list* sources the model path is ``passnet_dir / sample_path``.
    For *hf* sources ``sample_path`` is already absolute.
    """
    if source == "list":
        return f"{passnet_dir}/{sample_path}"
    return sample_path
