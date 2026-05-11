# Triton Kernel Extractor

A pipeline that compiles computational subgraphs through TorchInductor, filters
the results by kernel-level speedup, and extracts the autotuning-selected Triton
kernel source together with the corresponding PTX assembly from the inductor
compilation cache.

## Background

When `torch.compile` processes a model via the TorchInductor backend with
`TORCH_COMPILE_DEBUG=1`, the compiler produces a per-graph cache directory
containing:

- **`output_code.py`** — the generated Python wrapper that calls into Triton
  kernels via `async_compile.triton('kernel_name', '''...''')`.  The kernels
  appearing here are the final, autotuning-selected implementations adopted by
  the inductor scheduler.
- **`triton/0/{HASH}/`** — one directory per autotuning candidate
  configuration (varying `XBLOCK`, `YBLOCK`, `num_warps`, etc.), each holding
  the compiled artifacts (`.ptx`, `.cubin`, `.ttir`, `.llir`, `.source`,
  `.json`).  When autotuning explores N configurations for a kernel, N
  directories are created.
- **`*.best_config`** — a JSON file written by the Triton autotuner recording
  the winning configuration.  Its `triton_cache_hash` field maps back to one of
  the `triton/0/{HASH}/` directories.

This pipeline automates the full workflow: compile → filter → clean → extract →
pair, producing clean `(subgraph, triton_kernel, ptx)` triples ready for
downstream analysis.

## Pipeline Steps

The pipeline processes three dataset categories — `sole_op_subgraphs`,
`fusible_subgraphs`, and `typical_subgraphs` — executing five steps for each:

### Step 1: Multi-GPU Parallel Compilation

Compiles each subgraph sample using `graph_net_bench.torch.test_compiler
--kernel-time` in an isolated subprocess.  Samples are distributed across
available GPUs in round-robin fashion, with one `ProcessPoolExecutor` worker per
GPU.  Each subprocess receives a dedicated `CUDA_VISIBLE_DEVICES` and an
isolated `TORCHINDUCTOR_CACHE_DIR`.  Pass `--max-autotune` to enable Inductor's
`max_autotune` mode (via `torch.compile(mode="max-autotune")`), which activates
comprehensive autotuning including `max_autotune_gemm`,
`coordinate_descent_tuning`, and `epilogue_fusion`.

### Step 2: Speedup Filtering

Parses the `[Speedup][kernel]:` metric from each sample's compilation log (the
last occurrence is used).  Samples achieving a speedup ≥ 1.0 are moved to
`kept/`; the rest are moved to `discarded/`.

### Step 3: Temporary File Cleanup

Recursively removes `__pycache__/` directories, `*.pyc`, and `*.pyo` files from
the `kept/` tree to reduce storage footprint before extraction.

### Step 4: Kernel and PTX Extraction

For each kept sample that contains `original_graph/graph_hash.txt`:

1. Copies `original_graph/model.py` (the source subgraph) into the output.
2. Parses `output_code.py` to extract all Triton kernel definitions using a
   regex equivalent of the original Perl one-liner.
3. Writes each kernel source to `triton_kernel/{kernel_name}.py`.
4. Locates the corresponding PTX for each kernel by scanning `triton/0/` and
   disambiguating via `.best_config` when multiple autotuning candidates exist,
   then writes it to `ptx/{kernel_name}.ptx`.

Output is written atomically (`.tmp` directory + `rename`) so that an
interrupted run never leaves half-written data.

### Step 5: Empty Sample Cleanup

Removes output samples that contain `original_graph/` but no `triton_kernel/`
directory (i.e., samples where no Triton kernels were extracted).

## PTX Resolution Algorithm

Each Triton kernel may have been compiled under multiple autotuning
configurations.  The algorithm to locate the winning PTX is:

1. Scan `triton/0/*/` for directories containing `{kernel_name}.ptx`.
2. If exactly one candidate exists, use it directly (no autotuning was needed).
3. If multiple candidates exist, collect `triton_cache_hash` values from all
   `*.best_config` files in the sample, and select the candidate whose directory
   name matches one of these hashes.

This approach was validated on 125 kernels across 98 samples with a 100% match
rate.

## Output Structure

```
{output_dir}/{sample_name}/
    original_graph/
        model.py                           # source subgraph
    triton_kernel/
        triton_poi_fused_xxx_0.py          # Triton kernel source
        triton_poi_fused_yyy_1.py
    ptx/
        triton_poi_fused_xxx_0.ptx         # corresponding PTX assembly
        triton_poi_fused_yyy_1.ptx
```

## Cache Analysis

Analyzes an inductor cache directory post-hoc, available as the `analyze`
subcommand or triggered automatically by passing `--enable-cache-analysis` to
the `extract` subcommand.  Concatenates `test_compiler_log.log` files across
all sample states (root, kept, discarded), computes kernel and end-to-end
speedup distributions (mean, median, percentiles, threshold breakdowns), and
generates histogram, CDF, and optionally violin/ES(t) plots.  Output defaults
to `<cache_dir>_analysis/`.

## Usage

### Via the Bash Launcher

```bash
# Edit machine-specific paths in extract_triton_kernels.sh first, then:
bash tools/extract_triton_kernels.sh list            # auto-detect GPUs
bash tools/extract_triton_kernels.sh hf 0,2,5,7      # specify GPUs
```

### Via Python Directly

```bash
python3 -m tools.triton_kernel_extractor \
    --source list \
    --dataset-base-dir /data/passnet_dataset \
    --graphnet-dir /opt/GraphNet \
    --passnet-dir /opt/passnet \
    --passnet-hf-dir /opt/passnet/graphs/hf_subgraphs_v2 \
    --gpu-ids 0 2 5 7 \
    --max-autotune \
    --enable-cache-analysis

# Cache analysis can also be run standalone:
python3 -m tools.triton_kernel_extractor analyze <cache_dir> [--output-dir DIR]
```

### CLI Arguments

| Argument                   | Required | Description                                           |
|----------------------------|----------|-------------------------------------------------------|
| `--source`                 | Yes      | `list` (sample paths from text files) or `hf` (scan HuggingFace directories) |
| `--dataset-base-dir`       | Yes      | Root directory for cache and extraction output         |
| `--graphnet-dir`           | Yes      | Path to the GraphNet repository (for `graph_net_bench` on PYTHONPATH) |
| `--passnet-dir`            | Yes      | Root of the PassNet repository (model path prefix)    |
| `--passnet-hf-dir`         | No       | HuggingFace graph data directory; defaults to `{passnet-dir}/graphs/hf_subgraphs_v2` |
| `--gpu-ids`                | No       | GPU IDs for compilation; auto-detected when omitted    |
| `--max-autotune`           | No       | Enable Inductor max_autotune mode (`torch.compile(mode="max-autotune")`) |
| `--enable-cache-analysis`  | No       | Run cache analysis on each dataset after extraction    |

## Module Structure

```
triton_kernel_extractor/
    __init__.py              # package marker
    __main__.py              # CLI entry point (subcommands: extract, analyze)
    config.py                # PipelineConfig, DatasetDescriptor, constants
    sample_enumerator.py     # enumerate samples from "list" or "hf" sources
    compiler.py              # Step 1: multi-GPU parallel compilation
    speedup_filter.py        # Step 2: filter by kernel speedup
    temp_cleaner.py          # Step 3: remove __pycache__ / *.pyc / *.pyo
    kernel_extractor.py      # Step 4: extract Triton kernels and PTX
    empty_sample_cleaner.py  # Step 5: remove samples without Triton kernels
    pipeline.py              # orchestrate Steps 1–5 for all datasets
    cache_analyzer.py        # analyze cache: logs, statistics, plots
```

## Idempotency and Resume

Every step implements skip logic to support safe re-execution:

- **Compilation** skips samples whose log already contains `[Speedup][kernel]:`
  or that already exist under `kept/` or `discarded/`.
- **Filtering** skips samples already classified into `kept/` or `discarded/`.
- **Extraction** skips output samples that already exist in the output directory.
  Stale `.tmp` directories from prior interrupted runs are cleaned up
  automatically on startup.
