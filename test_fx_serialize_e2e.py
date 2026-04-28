import torch
import time
import sys
import os

# Add project root to path
sys.path.insert(0, "/ssd2/liangtai/ai4c")

# Mock pass source validator so we can use torch APIs in test pass files
import graph_net_bench.ast_util
graph_net_bench.ast_util.validate_pass_source = lambda src: []

from graph_net_bench.torch.backend.pass_mgr_fx_serialize import PassMgrFXSerializeBackend
import graph_net_bench.torch.backend.pass_mgr_backend as pmb


# ============================================================
# 1. Define a fused kernel (for testing: just calls add)
# ============================================================
def fused_add(x, y):
    """Replacement kernel: add."""
    return torch.add(x, y)


# ============================================================
# 2. Define test model with a simple add op
# ============================================================
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(128, 128))

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        z = torch.add(y, 1.0)
        return z


def test_correctness_and_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    x = torch.randn(64, 128, device=device, dtype=dtype)

    model = TestModel().to(device).eval()
    # Eager baseline
    with torch.no_grad():
        eager_out = model(x)

    # ============================================================
    # Build backend with ONE pass
    # ============================================================
    import tempfile
    import json
    tmpdir = tempfile.mkdtemp()
    input_dir = os.path.join(tmpdir, "input_pass")
    output_dir = os.path.join(tmpdir, "output_pass")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Write a single pass rule file
    pass_file = os.path.join(output_dir, "add_fuse.py")
    with open(pass_file, "w") as f:
        f.write("""
import torch

def pattern(a, b):
    return torch.add(a, b)

def replacement_func():
    return fused_add

def replacement_args(*args):
    return args

def fused_add(a, b):
    return torch.add(a, b)
""")

    # Write sorted lists
    with open(os.path.join(output_dir, "sorted_output_pass_rule_names.json"), "w") as f:
        json.dump(["add_fuse"], f)
    with open(os.path.join(input_dir, "sorted_input_pass_rule_names.json"), "w") as f:
        json.dump([], f)

    config = {
        "input_pass_rule_dir": input_dir,
        "output_pass_rule_dir": output_dir,
        "output_pass_pattern_limit": 10,
        "output_pass_replacement_func_limit": 10,
    }

    # Reset global singleton so PatternReplacementPass can set it
    pmb.g_replacement_func = None

    backend = PassMgrFXSerializeBackend(config)
    compiled_model = backend(model)

    # Warmup triggers torch.compile backend
    with torch.no_grad():
        for _ in range(3):
            compiled_model(x)

    # Now backend._optimized_gm should be the serialized module
    assert backend._optimized_gm is not None, "Serialization did not happen!"
    serialized = backend._optimized_gm

    print(f"[Test] Serialized module type: {type(serialized).__name__}")
    print(f"[Test] Serialized module class: {type(serialized).__module__}.{type(serialized).__name__}")

    # Verify it is NOT an FX GraphModule (true standalone)
    import torch.fx as fx
    is_fx_gm = isinstance(serialized, fx.GraphModule)
    print(f"[Test] Is FX GraphModule: {is_fx_gm}")
    assert not is_fx_gm, "Serialized module should be a plain nn.Module, not FX GraphModule"

    # Correctness check: use compiled_model so Dynamo handles flattened inputs
    with torch.no_grad():
        serialized_out = compiled_model(x)

    max_diff = torch.max(torch.abs(eager_out - serialized_out)).item()
    print(f"[Test] Correctness max_diff: {max_diff}")
    assert max_diff < 1e-5, f"Correctness failed: max_diff={max_diff}"

    # ============================================================
    # Benchmark: Eager vs torch.compile(serialized backend)
    # ============================================================
    warmup = 10
    trials = 50

    def benchmark(fn, label):
        for _ in range(warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(trials):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        avg_ms = (end - start) / trials * 1000
        print(f"[Benchmark] {label}: {avg_ms:.4f} ms")
        return avg_ms

    with torch.no_grad():
        eager_ms = benchmark(lambda: model(x), "Eager")
        compiled_ms = benchmark(lambda: compiled_model(x), "Compiled(serialized backend)")

    print(f"[Benchmark] Speedup Eager->Compiled: {eager_ms/compiled_ms:.2f}x")

    # Cleanup temp dirs
    import shutil
    shutil.rmtree(tmpdir)

    print("\n[Test] ALL PASSED!")


if __name__ == "__main__":
    test_correctness_and_benchmark()
