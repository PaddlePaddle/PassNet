"""Orchestrate the five-step extraction pipeline for all datasets."""

from __future__ import annotations

import logging
from pathlib import Path

from .compiler import compile_all_samples
from .config import (
    DatasetDescriptor,
    PipelineConfig,
    build_dataset_descriptors,
)
from .empty_sample_cleaner import clean_empty_kernel_samples
from .kernel_extractor import extract_triton_kernels
from .sample_enumerator import enumerate_hf_samples, enumerate_list_samples
from .speedup_filter import filter_samples_by_speedup
from .temp_cleaner import clean_temp_files

logger = logging.getLogger(__name__)


def _load_samples(
    config: PipelineConfig,
    dataset: DatasetDescriptor,
) -> list[str]:
    """Load the sample list for a single dataset from the appropriate source."""
    if config.source == "list":
        if dataset.graph_list_file is None:
            raise ValueError("graph_list_file must be set for 'list' source")
        if not dataset.graph_list_file.is_file():
            logger.error(
                "Graph list file not found: %s", dataset.graph_list_file
            )
            return []
        return enumerate_list_samples(dataset.graph_list_file)

    # source == "hf"
    if dataset.hf_subdir is None:
        raise ValueError("hf_subdir must be set for 'hf' source")
    return enumerate_hf_samples(config.graphnet_hf_dir, dataset.hf_subdir)


def run_dataset_pipeline(
    config: PipelineConfig,
    dataset: DatasetDescriptor,
    dataset_idx: int,
    total_datasets: int,
) -> None:
    """Execute all five pipeline steps for a single dataset."""
    logger.info("")
    logger.info("======================================================")
    logger.info(
        " Dataset [%d/%d]: %s", dataset_idx, total_datasets, dataset.name
    )

    samples = _load_samples(config, dataset)

    if config.source == "list":
        logger.info(" Graph list: %s", dataset.graph_list_file)
    else:
        logger.info(
            " HF dir:     %s/%s", config.graphnet_hf_dir, dataset.hf_subdir
        )

    logger.info(" Samples:    %d", len(samples))
    logger.info(" Cache dir:  %s", dataset.cache_dir)
    logger.info(" Output dir: %s", dataset.output_dir)
    logger.info(" GPUs:       %s", " ".join(str(g) for g in config.gpu_ids))
    logger.info("======================================================")

    if not samples:
        logger.error("No samples found for dataset: %s", dataset.name)
        return

    dataset.cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parallel compilation.
    num_gpus = len(config.gpu_ids)
    logger.info("")
    logger.info("=== Step 1: Parallel compilation (%d GPUs) ===", num_gpus)
    compile_all_samples(samples, config, dataset)

    # Step 2: Filter by speedup.
    logger.info("")
    logger.info("=== Step 2: Filter by speedup ===")
    filter_samples_by_speedup(dataset.cache_dir)

    # Step 3: Clean temp files.
    logger.info("")
    logger.info("=== Step 3: Clean temp files ===")
    clean_temp_files(dataset.cache_dir / "kept")

    # Step 4: Extract triton kernels.
    logger.info("")
    logger.info("=== Step 4: Extract autotuning-selected triton kernels and corresponding PTX ===")
    extract_triton_kernels(dataset.cache_dir, dataset.output_dir)

    # Step 5: Clean samples without triton kernels.
    logger.info("")
    logger.info("=== Step 5: Clean samples without triton kernels ===")
    clean_empty_kernel_samples(dataset.output_dir)


def run_pipeline(
    config: PipelineConfig,
    *,
    enable_cache_analysis: bool = False,
) -> None:
    """Run the full pipeline across all three dataset categories."""
    descriptors = build_dataset_descriptors(config)
    total = len(descriptors)

    for idx, dataset in enumerate(descriptors, 1):
        run_dataset_pipeline(config, dataset, idx, total)

    if enable_cache_analysis:
        from .cache_analyzer import analyze_cache

        for idx, dataset in enumerate(descriptors, 1):
            logger.info("")
            logger.info(
                "=== Cache analysis [%d/%d]: %s ===", idx, total, dataset.name
            )
            analysis_dir = dataset.cache_dir.parent / f"{dataset.cache_dir.name}_analysis"
            analyze_cache(dataset.cache_dir, analysis_dir)

    logger.info("")
    logger.info("All datasets processed.")
