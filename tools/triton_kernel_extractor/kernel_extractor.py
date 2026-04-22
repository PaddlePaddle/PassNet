"""Step 4: Extract triton kernel source code from ``output_code.py`` files."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Compiled regex that replaces the original perl one-liner:
#
#   perl -0777 -ne '
#       while (/async_compile\.triton\(\x27([^\x27]+)\x27,\s*\x27\x27\x27(.*?)\x27\x27\x27/gs) {
#           print "===KERNEL_NAME===$1\n$2\n===KERNEL_END===\n";
#       }'
#
# Captures: group(1) = kernel name, group(2) = kernel source code.
_TRITON_KERNEL_PATTERN = re.compile(
    r"async_compile\.triton\('([^']+)',\s*'''(.*?)'''",
    re.DOTALL,
)


def extract_kernels_from_file(
    output_code_path: Path,
) -> list[tuple[str, str]]:
    """Parse an ``output_code.py`` and return ``(name, source)`` pairs.

    The file is read entirely into memory (``output_code.py`` files produced by
    TorchInductor are typically well under 1 MB).
    """
    content = output_code_path.read_text(encoding="utf-8", errors="replace")
    return _TRITON_KERNEL_PATTERN.findall(content)


def extract_triton_kernels(
    cache_dir: Path,
    output_dir: Path,
) -> tuple[int, int, int, int]:
    """Walk kept samples, extract triton kernels, and write paired output.

    For every kept sample that contains ``original_graph/graph_hash.txt``:

    1. Copy ``original_graph/model.py`` into the output.
    2. Parse every ``output_code.py`` found in the sample tree.
    3. Write each extracted kernel to ``triton_kernel/{name}.py``.

    The output uses an atomic ``.tmp`` + ``rename`` pattern so that an
    interrupted run never leaves a half-written sample directory.

    Returns:
        ``(processed_files, total_kernels, copied_graphs, skip_count)``
    """
    kept_dir = cache_dir / "kept"
    if not kept_dir.is_dir():
        logger.error("Kept directory does not exist: %s", kept_dir)
        return 0, 0, 0, 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale .tmp directories from a previous interrupted run.
    for stale in output_dir.iterdir():
        if stale.is_dir() and stale.name.endswith(".tmp"):
            shutil.rmtree(stale, ignore_errors=True)

    # Collect eligible samples (must contain original_graph/graph_hash.txt).
    eligible: list[Path] = [
        d
        for d in sorted(kept_dir.iterdir())
        if d.is_dir() and (d / "original_graph" / "graph_hash.txt").is_file()
    ]
    total = len(eligible)

    processed_files = 0
    total_kernels = 0
    copied_graphs = 0
    skip_count = 0

    for idx, graph_dir in enumerate(eligible, 1):
        graph_name = graph_dir.name
        dest_graph_dir = output_dir / graph_name

        # Resume: skip if the final output already exists.
        if dest_graph_dir.exists():
            skip_count += 1
            continue

        logger.info("[%d/%d] Extracting: %s", idx, total, graph_name)

        # Write to a temporary directory; rename atomically on success.
        tmp_dir = dest_graph_dir.with_name(f"{graph_name}.tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        # Copy original model source when available.
        model_src = graph_dir / "original_graph" / "model.py"
        if model_src.is_file():
            og_dir = tmp_dir / "original_graph"
            og_dir.mkdir()
            shutil.copy2(str(model_src), str(og_dir / "model.py"))

        # Find and process all output_code.py files within the sample.
        for output_code_path in sorted(graph_dir.rglob("output_code.py")):
            processed_files += 1
            kernels = extract_kernels_from_file(output_code_path)
            if not kernels:
                continue

            triton_dir = tmp_dir / "triton_kernel"
            triton_dir.mkdir(exist_ok=True)

            for name, source in kernels:
                # Strip trailing whitespace then add exactly one newline,
                # matching the bash `printf '%s\n'` semantics.
                (triton_dir / f"{name}.py").write_text(
                    source.rstrip() + "\n", encoding="utf-8"
                )
                total_kernels += 1

        # Atomic completion: rename .tmp → final (same filesystem guarantees
        # a single rename(2) syscall).
        tmp_dir.rename(dest_graph_dir)
        copied_graphs += 1

    logger.info(
        "Extraction: %d files, %d kernels, %d graphs, %d skipped (total: %d)",
        processed_files,
        total_kernels,
        copied_graphs,
        skip_count,
        total,
    )
    logger.info("Output:     %s", output_dir)
    return processed_files, total_kernels, copied_graphs, skip_count
