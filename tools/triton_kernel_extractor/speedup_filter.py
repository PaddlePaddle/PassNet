"""Step 2: Partition compiled samples into *kept* and *discarded* by speedup."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from .config import (
    RESERVED_DIR_NAMES,
    RESERVED_DIR_PREFIX,
    SPEEDUP_KERNEL_PATTERN,
    SPEEDUP_THRESHOLD,
)

logger = logging.getLogger(__name__)

_SPEEDUP_RE = re.compile(SPEEDUP_KERNEL_PATTERN)


def _parse_kernel_speedup(log_file: Path) -> float | None:
    """Extract the last ``[Speedup][kernel]:`` value from a compiler log.

    Returns ``None`` when the log does not exist or contains no speedup line.
    """
    if not log_file.is_file():
        return None
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    last_match: re.Match[str] | None = None
    for m in _SPEEDUP_RE.finditer(content):
        last_match = m

    if last_match is None:
        return None

    try:
        return float(last_match.group(1))
    except ValueError:
        return None


def _is_sample_dir(name: str) -> bool:
    """Return True if *name* is a real sample directory (not a reserved one)."""
    return name not in RESERVED_DIR_NAMES and not name.startswith(
        RESERVED_DIR_PREFIX
    )


def filter_samples_by_speedup(cache_dir: Path) -> tuple[int, int, int]:
    """Move compiled samples into ``kept/`` or ``discarded/`` sub-directories.

    A sample is *kept* when the last ``[Speedup][kernel]:`` value in its
    ``test_compiler_log.log`` is >= ``SPEEDUP_THRESHOLD``.  Samples that
    have already been classified are silently skipped.

    Returns:
        A tuple of ``(kept_count, discarded_count, skip_count)``.
    """
    kept_dir = cache_dir / "kept"
    discarded_dir = cache_dir / "discarded"
    kept_dir.mkdir(parents=True, exist_ok=True)
    discarded_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidate directories (snapshot the listing to avoid mutating
    # the iterator while moving entries).
    candidates: list[Path] = [
        d
        for d in sorted(cache_dir.iterdir())
        if d.is_dir() and _is_sample_dir(d.name)
    ]
    total = len(candidates)

    kept_count = 0
    discarded_count = 0
    skip_count = 0

    for idx, graph_dir in enumerate(candidates, 1):
        graph_name = graph_dir.name

        # Skip if already classified in a previous (possibly interrupted) run.
        if (kept_dir / graph_name).exists() or (
            discarded_dir / graph_name
        ).exists():
            skip_count += 1
            continue

        log_file = graph_dir / "test_compiler_log.log"
        speedup = _parse_kernel_speedup(log_file)
        should_keep = speedup is not None and speedup >= SPEEDUP_THRESHOLD

        if should_keep:
            shutil.move(str(graph_dir), str(kept_dir / graph_name))
            kept_count += 1
        else:
            shutil.move(str(graph_dir), str(discarded_dir / graph_name))
            discarded_count += 1

        label = "KEPT" if should_keep else "DISCARDED"
        logger.info("[%d/%d] %s: %s", idx, total, label, graph_name)

    logger.info(
        "Filter: %d kept, %d discarded, %d skipped (total: %d)",
        kept_count,
        discarded_count,
        skip_count,
        total,
    )
    return kept_count, discarded_count, skip_count
