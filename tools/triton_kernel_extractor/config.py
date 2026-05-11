"""Pipeline configuration types and dataset descriptor construction."""

from __future__ import annotations

import dataclasses
from pathlib import Path

# The three dataset categories processed by the pipeline.
DATASET_NAMES: tuple[str, ...] = (
    "sole_op_subgraphs",
    "fusible_subgraphs",
    "typical_subgraphs",
)

# Log pattern emitted by the external GraphNet test_compiler with --kernel-time.
# Used in speedup_filter to decide whether to keep or discard a compiled sample.
SPEEDUP_KERNEL_PATTERN = r"\[Speedup\]\[kernel\]:\s*([\d.]+)"

# Subdirectory names reserved for internal bookkeeping inside the cache directory.
# These are skipped when iterating over sample directories.
RESERVED_DIR_NAMES = frozenset({"kept", "discarded"})

# Prefix used by temporary pipeline artifacts (chunk files, worker logs, sample
# lists).  Directories whose name starts with this prefix are skipped during
# sample iteration.
RESERVED_DIR_PREFIX = "_"

# Minimum kernel speedup required to keep a compiled sample.
SPEEDUP_THRESHOLD = 1.0


def is_sample_dir(name: str) -> bool:
    """Return True if *name* is a real sample directory, not a reserved one.

    Filters out ``kept``, ``discarded``, and directories starting with ``_``
    (temporary pipeline artifacts such as chunk files and worker logs).
    """
    if name in RESERVED_DIR_NAMES:
        return False
    if name.startswith(RESERVED_DIR_PREFIX):
        return False
    return True


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Immutable top-level configuration for the entire pipeline run."""

    source: str  # "list" or "hf"
    gpu_ids: list[int]
    dataset_base_dir: Path
    graphnet_dir: Path
    passnet_dir: Path
    passnet_hf_dir: Path
    max_autotune: bool = False


@dataclasses.dataclass(frozen=True)
class DatasetDescriptor:
    """Describes one of the three dataset categories to be processed."""

    name: str
    cache_dir: Path
    output_dir: Path
    # Only populated when source == "list".
    graph_list_file: Path | None
    # Only populated when source == "hf".
    hf_subdir: str | None


def build_dataset_descriptors(
    config: PipelineConfig,
) -> list[DatasetDescriptor]:
    """Build the list of dataset descriptors from the pipeline configuration.

    The mapping mirrors the original bash arrays ``DATASET_NAMES``,
    ``CACHE_DIRS``, ``GRAPH_LIST_FILES``, and ``HF_SUBDIRS``.
    """
    if config.source == "list":
        dataset_dir = config.dataset_base_dir / "GitHubV2"
        graph_list_files = [
            config.dataset_base_dir / "hf_sole_op_samples_v2_all_expanded.txt",
            config.dataset_base_dir / "hf_fusible_samples_v2_all_expanded.txt",
            config.dataset_base_dir / "hf_typical_samples_v2_all_expanded.txt",
        ]
    elif config.source == "hf":
        dataset_dir = config.dataset_base_dir / "Huggingface"
        graph_list_files = [None, None, None]
    else:
        raise ValueError(
            f"Invalid source {config.source!r}. Must be 'list' or 'hf'."
        )

    if config.source == "hf":
        hf_subdirs = list(DATASET_NAMES)
    else:
        hf_subdirs = [None, None, None]

    descriptors: list[DatasetDescriptor] = []
    for name, graph_list_file, hf_subdir in zip(
        DATASET_NAMES, graph_list_files, hf_subdirs, strict=True
    ):
        cache_dir = dataset_dir / f"{name}_inductor_dump"
        output_dir = dataset_dir / f"{name}_inductor_dump_subgraph_triton_kernel_pair"
        descriptors.append(
            DatasetDescriptor(
                name=name,
                cache_dir=cache_dir,
                output_dir=output_dir,
                graph_list_file=graph_list_file,
                hf_subdir=hf_subdir,
            )
        )

    return descriptors
