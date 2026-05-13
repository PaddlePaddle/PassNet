# PassBench Evaluation Pipeline

The PassNet evaluation pipeline works as follows:

1. **Analyze computation graph**: Read the target subgraph's `model.py` and `weight_meta.py`
2. **Generate optimization pass**: Create pattern matching rules and replacement functions
3. **Pass matching and replacement**: The `pass_mgr` compiler applies the generated pass
4. **Correctness verification**: Validate numerical consistency between the optimized and original kernels
5. **Performance benchmarking**: Measure speedup and output evaluation results

The concrete steps are: **write a pass → place it in `pass_dir/` → run `entry.sh`**.

## Pass File Format

A pass file is a Python module placed in the sample's `pass_dir/`. It must expose three module-level functions:

| Function | Description |
|----------|-------------|
| `pattern(*args)` | Describes the target computation subgraph using PyTorch ops; used as the matching template against the FX graph |
| `replacement_args(*args)` | Maps the matched pattern inputs to the arguments forwarded to the replacement kernel |
| `replacement_func()` | Returns the optimized kernel wrapper (must return a stable module-level function, not a nested `def` or `lambda`) |

The pass file is typically structured as:

```
MyPass.py
├── def pattern(...)          # subgraph to match
├── def replacement_args(...) # argument mapping
├── @triton.jit kernel        # optimized Triton kernel
├── @torch.fx.wrap wrapper    # kernel wrapper
└── def replacement_func()    # returns the wrapper
```

## Placing the Pass in `pass_dir/`

Place the pass file in the sample's `pass_dir/` together with a `sorted_output_pass_rule_names.json` that declares the loading order (file stem names, without `.py`):

```
samples/hf_subgraphs_v2/fusible_subgraphs/c3/88/<hash>/
├── graphs/
├── graph_list.txt
├── entry.sh
└── pass_dir/
    ├── MyPass.py
    └── sorted_output_pass_rule_names.json
```

`sorted_output_pass_rule_names.json`:

```json
["MyPass"]
```

Multiple passes are supported; list them in priority order:

```json
["PassA", "PassB"]
```

## Running a Single Sample

```bash
bash samples/hf_subgraphs_v2/fusible_subgraphs/c3/88/<hash>/entry.sh
```

`entry.sh` automatically:

1. Checks that `pass_bench` is on `PYTHONPATH`
2. Loads passes from `pass_dir/` and applies `pass_mgr` to match and replace subgraphs
3. Verifies numerical correctness (eager vs. optimized output within tolerance)
4. Benchmarks performance (25 warmup + 100 trial runs on GPU)
5. Writes results to `/tmp/workspace_pass_bench_test/aggregated_score.json`

A successful run prints:

```
[PassMgrBackend] Loaded 1 passes: ['MyPass']
[PassMgrBackend] Applied 1 replacements with MyPass.
Has Any pass matched? [True]
...
{"es": 1.0, "speedup": 1.35, ...}
```

If the pass does not match, the run exits early:

```
Has Any pass matched? [False]
Pass testing early exits on pass mismatch.
```
