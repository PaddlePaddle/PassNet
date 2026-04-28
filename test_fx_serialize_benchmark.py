import torch
import torch.nn.functional as F
import time
import sys
import os

sys.path.insert(0, "/ssd2/liangtai/ai4c")

# Mock pass source validator
import graph_net_bench.ast_util
graph_net_bench.ast_util.validate_pass_source = lambda src: []

from graph_net_bench.torch.backend.pass_mgr_fx_serialize import PassMgrFXSerializeBackend
from graph_net_bench.torch.backend.pass_mgr_backend import PassMgrBackend
from graph_net_bench.torch.override_dispatch_flag import global_override_dispatch
import graph_net_bench.torch.backend.pass_mgr_backend as pmb


# ============================================================
# Synthetic fused kernel: just calls the ops inline
# In a real pass this would be a custom CUDA kernel.
# ============================================================
def fused_matmul_add_relu(x, weight, bias):
    y = torch.matmul(x, weight)
    z = torch.add(y, bias)
    return torch.relu(z)


class TestModel(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim, dim))
        self.bias = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        z = torch.add(y, self.bias)
        return torch.relu(z)


def build_backend(backend_name, tmpdir):
    input_dir = os.path.join(tmpdir, "input_pass")
    output_dir = os.path.join(tmpdir, "output_pass")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    pass_file = os.path.join(output_dir, "matmul_add_relu.py")
    with open(pass_file, "w") as f:
        f.write("""
import torch

def pattern(x, weight, bias):
    y = torch.matmul(x, weight)
    z = torch.add(y, bias)
    return torch.relu(z)

def replacement_func():
    return fused_matmul_add_relu

def replacement_args(*args):
    return args

def fused_matmul_add_relu(x, weight, bias):
    y = torch.matmul(x, weight)
    z = torch.add(y, bias)
    return torch.relu(z)
""")

    import json
    with open(os.path.join(output_dir, "sorted_output_pass_rule_names.json"), "w") as f:
        json.dump(["matmul_add_relu"], f)
    with open(os.path.join(input_dir, "sorted_input_pass_rule_names.json"), "w") as f:
        json.dump([], f)

    config = {
        "input_pass_rule_dir": input_dir,
        "output_pass_rule_dir": output_dir,
        "output_pass_pattern_limit": 10,
        "output_pass_replacement_func_limit": 10,
    }

    pmb.g_replacement_func = None

    if backend_name == "pass_mgr":
        return PassMgrBackend(config)
    elif backend_name == "pass_mgr_fx_serialize":
        return PassMgrFXSerializeBackend(config)
    else:
        raise ValueError(backend_name)


def benchmark(fn, label, warmup=10, trials=50, device="cuda"):
    for _ in range(warmup):
        with global_override_dispatch(False):
            fn()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(trials):
        with global_override_dispatch(False):
            fn()
    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) / trials * 1000
    print(f"[Benchmark] {label}: {avg_ms:.4f} ms")
    return avg_ms


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    dim = 512
    batch = 256
    x = torch.randn(batch, dim, device=device, dtype=dtype)

    model = TestModel(dim).to(device).eval()

    # Eager baseline
    with torch.no_grad():
        eager_out = model(x)
    eager_ms = benchmark(lambda: model(x), "Eager", device=device)

    import tempfile
    tmpdir = tempfile.mkdtemp()

    # pass_mgr backend
    backend_pm = build_backend("pass_mgr", tmpdir)
    compiled_pm = backend_pm(model)
    with torch.no_grad():
        with global_override_dispatch(False):
            for _ in range(3):
                compiled_pm(x)
            pm_out = compiled_pm(x)
    pm_ms = benchmark(lambda: compiled_pm(x), "pass_mgr", device=device)

    max_diff = torch.max(torch.abs(eager_out - pm_out)).item()
    print(f"[Correctness] pass_mgr max_diff: {max_diff}")

    # pass_mgr_fx_serialize backend
    backend_ser = build_backend("pass_mgr_fx_serialize", tmpdir)
    compiled_ser = backend_ser(model)
    with torch.no_grad():
        with global_override_dispatch(False):
            for _ in range(3):
                compiled_ser(x)
            ser_out = compiled_ser(x)
    ser_ms = benchmark(lambda: compiled_ser(x), "pass_mgr_fx_serialize", device=device)

    max_diff = torch.max(torch.abs(eager_out - ser_out)).item()
    print(f"[Correctness] pass_mgr_fx_serialize max_diff: {max_diff}")

    print(f"\n[Summary] Eager={eager_ms:.4f}ms  pass_mgr={pm_ms:.4f}ms  serialize={ser_ms:.4f}ms")
    print(f"[Summary] pass_mgr vs eager: {eager_ms/pm_ms:.2f}x")
    print(f"[Summary] serialize vs eager: {eager_ms/ser_ms:.2f}x")
    print(f"[Summary] serialize vs pass_mgr: {pm_ms/ser_ms:.2f}x")

    import shutil
    shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
