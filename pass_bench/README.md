# PassBench Evaluation Pipeline

**[English](README.md)** | **[中文](README_cn.md)**

PassNet evaluation pipeline:

1. [Analyze Computation Graph](#1-analyze-computation-graph)
2. [Generate Optimization Pass](#2-generate-optimization-pass)
3. [Pass Matching and Replacement](#3-pass-matching-and-replacement)
4. [Correctness Verification](#4-correctness-verification)
5. [Performance Benchmarking](#5-performance-benchmarking)

---

## 1. Analyze Computation Graph

Each sample directory contains one or more computation graphs under `graphs/`. Each graph is defined by these files:

| File | Description |
|------|-------------|
| `model.py` | PyTorch FX graph definition — the operators and their connections |
| `weight_meta.py` | Shapes, dtypes, and device info of weight tensors |
| `input_meta.py` | Shapes and dtypes of input tensors |
| `input_tensor_constraints.py` | Value range constraints for inputs |
| `graph_net.json` | Serialized graph structure |

A sample may contain multiple variants of the same subgraph for different dtypes (e.g. `float32`, `float16`, `bfloat16`) and different batch sizes. The evaluation runs the pass against all variants.

```
sample/
└── graphs/
    └── hf_subgraphs_v2/fusible_subgraphs/
        ├── float16/1/.../<subgraph_name>/
        │   ├── model.py
        │   ├── weight_meta.py
        │   ├── input_meta.py
        │   └── ...
        ├── float32/1/.../<subgraph_name>/
        └── bfloat16/1/.../<subgraph_name>/
```

The goal of this step is to identify the target computation pattern — which operators appear, in what order, and with what tensor shapes and dtypes.

---

## 2. Generate Optimization Pass

A pass file is a Python module placed in `pass_dir/`. It tells `pass_mgr` which subgraph pattern to match and what optimized kernel to replace it with.

### Pass file format

A pass file must expose three module-level functions:

| Function | Description |
|----------|-------------|
| `pattern(*args)` | Describes the target subgraph using PyTorch ops; `pass_mgr` uses this as the matching template against the FX graph |
| `replacement_args(*args)` | Maps matched pattern inputs to the arguments forwarded to the replacement kernel |
| `replacement_func()` | Returns the optimized kernel wrapper — must return a stable module-level function, not a nested `def` or `lambda` |

Typical pass file structure:

```
MyPass.py
├── def pattern(...)           # subgraph to match (PyTorch ops)
├── def replacement_args(...)  # argument remapping
├── @triton.jit kernel         # optimized Triton kernel implementation
├── @torch.fx.wrap wrapper     # kernel wrapper callable from FX graph
└── def replacement_func()     # returns the wrapper
```

### Placing the pass

Place the pass file in the sample's `pass_dir/` alongside a `sorted_output_pass_rule_names.json` that declares the loading order (file stem names, without `.py`):

```
sample/
└── pass_dir/
    ├── MyPass.py
    └── sorted_output_pass_rule_names.json   # ["MyPass"]
```

Multiple passes are supported; list them in priority order:

```json
["PassA", "PassB"]
```

---

## 3. Pass Matching and Replacement

`entry.sh` invokes `pass_bench.torch.test_compiler` with `--compiler pass_mgr`. The `PassMgrBackend` loads all pass files from `pass_dir/`, then for each graph:

1. Traces the model with `torch.compile` to obtain the FX graph
2. Uses `SubgraphMatcher` to find all occurrences of the `pattern` subgraph
3. Replaces each match with the kernel wrapper returned by `replacement_func()`
4. Recompiles the modified graph

Log output when matching succeeds:

```
[PassMgrBackend] Loaded 1 passes: ['MyPass']
[PassMgrBackend] Applied 1 replacements with MyPass.
```

If the pattern does not match any subgraph, `pass_mgr` raises an error and the run exits early:

```
[PassMgrBackend] Pass MyPass failed to match.
Has Any pass matched? [False]
Pass testing early exits on pass mismatch.
```

When pass mismatch occurs, the sample receives the minimum score (`ES(t) = 0.1` for all tolerance levels).

---

## 4. Correctness Verification

After the optimized graph is compiled, `test_compiler` runs both the original eager model and the compiled model on the same inputs, then compares outputs using `torch.allclose` across a sweep of tolerance levels.

### Dtype-specific precision thresholds

Each dtype has a fixed `(rtol, atol)` precision threshold used as the baseline correctness criterion:

| dtype | rtol | atol |
|-------|------|------|
| float32 | 1.3E-06 | 1.00E-05 |
| float16 | 1.00E-03 | 1.00E-05 |
| bfloat16 | 1.60E-02 | 1.00E-05 |

In the log, each correctness check is keyed as `[all_close_atol_<atol>_rtol_<rtol>]`, so the dtype-baseline checks appear as:

```
[Correctness][all_close_atol_1.00E-05_rtol_1.30E-06]: 1   # float32
[Correctness][all_close_atol_1.00E-05_rtol_1.00E-03]: 1   # float16
[Correctness][all_close_atol_1.00E-05_rtol_1.60E-02]: 1   # bfloat16
...
[Correctness][max_diff]: 0.0001220703125
[Correctness][mean_diff]: 1.862645149230957e-09
```

Dtype consistency is also verified:

```
[Datatype][eager]: bfloat16
[Datatype][compiled]: bfloat16
[DataType] eager:['bfloat16'] compiled:['bfloat16'] match:True
```

A graph is marked `success` only if both dtype and correctness checks pass:

```
[Result] status: success
```

Otherwise it is marked `failed`.

---

## 5. Performance Benchmarking

For each graph that passes correctness, `test_compiler` benchmarks both the eager and compiled models:

- **Warmup**: 25 runs (not measured)
- **Trials**: 100 timed runs, recording both end-to-end (`e2e`) and GPU-only (`gpu`) latency per run

```
[Profiling] Using device: cuda NVIDIA A30, warm up 25, trials 100
Trial 1: e2e=0.314 ms, gpu=0.260 ms
Trial 2: e2e=0.365 ms, gpu=0.313 ms
...
[Performance][eager]:    {"e2e": {"median": 0.264, ...}, "gpu": {"median": 0.230, ...}}
[Performance][compiled]: {"e2e": {"median": 0.297, ...}, "gpu": {"median": 0.259, ...}}
[Speedup][e2e]: 0.889
[Speedup][gpu]: 0.888
```

Speedup is computed as `eager_median / compiled_median`. A speedup > 1 means the optimized kernel is faster.

### Score aggregation

After all graphs in the sample are evaluated, `pass_bench.aggregate_es_scores` computes **ES(t)** — the primary metric — for each tolerance level `t`:

- For each graph, derive a *rectified speedup*:
  - If the graph is correct at `t=1` and has speedup `s`: rectified speedup = `s` (if `s ≥ 1`) or `s²` (penalizes slowdown)
  - If the graph fails correctness or pass matching: rectified speedup = `b = 0.1` (baseline penalty)
  - If the graph fails at `t=1` but the failure type is tolerated at tolerance `t`: rectified speedup = 1
- **ES(t)** = geometric mean of rectified speedups across all graphs in the sample

```
  - ESt=0.100 for tolerance=-10.
  - ESt=0.100 for tolerance=-5.
  - ESt=0.912 for tolerance=1.
  - ESt=0.912 for tolerance=2.
  ...
aggregated_speedup=0.912
Result is saved to /tmp/workspace_pass_bench_test/aggregated_score.json
```

The final result is written to `aggregated_score.json`.

---

## Running the Evaluation

```bash
# single sample
bash samples/<type>/<hash>/entry.sh

# batch evaluation
SAMPLE_LIST="sample_lists/eval/hf_fusible_eval_samples_v2.txt"
LOG_FILE="eval.log"
> "$LOG_FILE"
idx=0; total=$(grep -c . "$SAMPLE_LIST")

while IFS= read -r sample_path; do
    [ -z "$sample_path" ] && continue
    idx=$((idx + 1))
    echo "===== [$idx/$total] $(basename "$sample_path") =====" | tee -a "$LOG_FILE"
    bash "$sample_path/entry.sh" >> "$LOG_FILE" 2>&1 && \
        echo "OK" | tee -a "$LOG_FILE" || \
        echo "FAILED" | tee -a "$LOG_FILE"
done < "$SAMPLE_LIST"

echo "===== Total: $total =====" | tee -a "$LOG_FILE"
```
