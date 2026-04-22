"""Step 4: Extract autotuning-selected triton kernels and corresponding PTX."""

from __future__ import annotations

import json
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


def _collect_best_config_hashes(graph_dir: Path) -> set[str]:
    """Gather all autotuning-selected cache hashes from a sample directory.

    TorchInductor writes ``.best_config`` JSON files (one per autotuned kernel)
    in 2-char prefix subdirectories of the sample cache.  Each file contains a
    ``triton_cache_hash`` field identifying the winning configuration among
    multiple compiled candidates in ``triton/0/``.

    This function is called once per sample and the result is reused for every
    kernel in that sample.
    """
    hashes: set[str] = set()
    for bc_path in graph_dir.rglob("*.best_config"):
        try:
            data = json.loads(bc_path.read_text(encoding="utf-8"))
            cache_hash = data.get("triton_cache_hash")
            if cache_hash:
                hashes.add(cache_hash)
        except (OSError, json.JSONDecodeError, KeyError):
            logger.debug("Skipping malformed .best_config: %s", bc_path)
    return hashes


def _find_best_ptx(
    graph_dir: Path,
    kernel_name: str,
    best_hashes: set[str],
) -> str | None:
    """Locate the corresponding PTX for a given kernel via autotuning results.

    The inductor cache compiles each triton kernel into one or more candidate
    configurations under ``triton/0/{HASH}/``.  When autotuning runs, multiple
    candidate directories exist for the same kernel and a ``.best_config`` file
    records the winning ``triton_cache_hash``.

    Resolution strategy:
      - 0 candidates → return ``None`` (no PTX compiled for this kernel).
      - 1 candidate  → return its PTX (no disambiguation needed).
      - N candidates → intersect directory names with *best_hashes*; the match
        identifies the autotuning winner.
    """
    triton_base = graph_dir / "triton" / "0"
    if not triton_base.is_dir():
        return None

    # Collect all triton/0/{hash}/ dirs that contain this kernel's PTX.
    ptx_filename = f"{kernel_name}.ptx"
    candidates: list[Path] = [
        ptx_file
        for hash_dir in triton_base.iterdir()
        if hash_dir.is_dir()
        for ptx_file in [hash_dir / ptx_filename]
        if ptx_file.is_file()
    ]

    if not candidates:
        logger.debug("No PTX found for kernel %s in %s", kernel_name, graph_dir.name)
        return None

    if len(candidates) == 1:
        return candidates[0].read_text(encoding="utf-8", errors="replace")

    # Multiple candidates: pick the one whose parent dir matches a best_config hash.
    for ptx_path in candidates:
        if ptx_path.parent.name in best_hashes:
            return ptx_path.read_text(encoding="utf-8", errors="replace")

    # Fallback: no .best_config match (should not happen based on validation).
    logger.warning(
        "Multiple PTX candidates for %s but no .best_config match in %s",
        kernel_name,
        graph_dir.name,
    )
    return None


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
) -> tuple[int, int, int, int, int]:
    """Walk kept samples, extract autotuning-selected triton kernels and corresponding PTX.

    For every kept sample that contains ``original_graph/graph_hash.txt``:

    1. Copy ``original_graph/model.py`` into the output.
    2. Parse every ``output_code.py`` found in the sample tree.
    3. Write each extracted kernel to ``triton_kernel/{name}.py``.
    4. Locate the corresponding PTX for each kernel and write it to
       ``ptx/{name}.ptx``.

    The output uses an atomic ``.tmp`` + ``rename`` pattern so that an
    interrupted run never leaves a half-written sample directory.

    Returns:
        ``(processed_files, total_kernels, total_ptx, copied_graphs, skip_count)``
    """
    kept_dir = cache_dir / "kept"
    if not kept_dir.is_dir():
        logger.error("Kept directory does not exist: %s", kept_dir)
        return 0, 0, 0, 0, 0

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
    total_ptx = 0
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

        # Pre-collect autotuning best-config hashes once per sample.
        best_hashes = _collect_best_config_hashes(graph_dir)

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

                # Locate and write the corresponding PTX for this kernel.
                ptx_content = _find_best_ptx(graph_dir, name, best_hashes)
                if ptx_content is not None:
                    ptx_dir = tmp_dir / "ptx"
                    ptx_dir.mkdir(exist_ok=True)
                    (ptx_dir / f"{name}.ptx").write_text(
                        ptx_content, encoding="utf-8"
                    )
                    total_ptx += 1

        # Atomic completion: rename .tmp → final (same filesystem guarantees
        # a single rename(2) syscall).
        tmp_dir.rename(dest_graph_dir)
        copied_graphs += 1

    logger.info(
        "Extraction: %d files, %d kernels, %d ptx, %d graphs, %d skipped (total: %d)",
        processed_files,
        total_kernels,
        total_ptx,
        copied_graphs,
        skip_count,
        total,
    )
    logger.info("Output:     %s", output_dir)
    return processed_files, total_kernels, total_ptx, copied_graphs, skip_count
