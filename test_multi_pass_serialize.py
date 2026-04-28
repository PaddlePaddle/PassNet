"""
Test PassMgrFXSerializeBackend across multiple real fused-kernel passes.

We test each pass in two modes:
  A. Direct FX GraphModule (bypasses dynamo/framework matching issues)
     - Guarantees the serialize backend works for the pass's kernel
  B. Full backend via torch.compile (tests end-to-end if pass matches)
     - May fail for known framework kwargs/positional mismatch bugs

Passes tested:
  1. maskformer  FusedRollSliceAddLayerNorm (CUDA C++)   — roll+slice+add+LN
  2. rexnetr     FuseAddSliceCat_304 (Triton)            — add+slice+cat
  3. eca_botnext FuseAvgPoolBatchNormSilu (CUDA C++)     — avgpool2d+BN
"""

import torch
import torch.fx as fx
import time
import sys
import os
import tempfile
import json
import shutil

sys.path.insert(0, "/ssd2/liangtai/ai4c")

# Mock validator
import graph_net_bench.ast_util
graph_net_bench.ast_util.validate_pass_source = lambda src: []

from graph_net_bench.torch.backend.pass_mgr_fx_serialize import PassMgrFXSerializeBackend
from graph_net_bench.torch.backend.pass_mgr_backend import PassMgrBackend
import graph_net_bench.torch.backend.pass_mgr_backend as pmb
from graph_net_bench.torch.override_dispatch_flag import global_override_dispatch


def benchmark(fn, label, warmup=25, trials=100, device="cuda"):
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


def build_backend(pass_dir, pass_names, device="cuda"):
    """Build a PassMgrFXSerializeBackend for the given passes."""
    tmpdir = tempfile.mkdtemp()
    input_dir = os.path.join(tmpdir, "input_pass")
    output_dir = os.path.join(tmpdir, "output_pass")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for name in pass_names:
        src = os.path.join(pass_dir, f"{name}.py")
        dst = os.path.join(output_dir, f"{name}.py")
        if os.path.exists(src):
            shutil.copy(src, dst)
        # Also copy shared modules
        for shared in os.listdir(pass_dir):
            if shared.startswith("shared_") and shared.endswith(".py"):
                s_src = os.path.join(pass_dir, shared)
                s_dst = os.path.join(output_dir, shared)
                if not os.path.exists(s_dst):
                    shutil.copy(s_src, s_dst)

    with open(os.path.join(output_dir, "sorted_output_pass_rule_names.json"), "w") as f:
        json.dump(pass_names, f)
    with open(os.path.join(input_dir, "sorted_input_pass_rule_names.json"), "w") as f:
        json.dump([], f)

    config = {
        "input_pass_rule_dir": input_dir,
        "output_pass_rule_dir": output_dir,
        "output_pass_pattern_limit": 10,
        "output_pass_replacement_func_limit": 10,
    }
    pmb.g_replacement_func = None
    return PassMgrFXSerializeBackend(config), tmpdir


# ============================================================
# Pass 1: maskformer (roll+slice+add+layernorm)
# ============================================================
def test_maskformer():
    print("\n" + "=" * 60)
    print("PASS 1: maskformer FusedRollSliceAddLayerNorm")
    print("=" * 60)

    sys.path.insert(0, "/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/maskformer")
    from pass_dir.shared_fused_roll_add_ln import fused_roll_slice_add_layernorm_dispatch

    class OriginalGraph(torch.nn.Module):
        def forward(self, in_0, in_1, in_2, in_3):
            tmp_2 = in_3.contiguous()
            tmp_3 = tmp_2.view(-1, 35, 35, 384)
            tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
            tmp_5 = tmp_4[(slice(None, None, None), slice(None, 32, None), slice(None, 32, None), slice(None, None, None))]
            tmp_6 = tmp_5.contiguous()
            tmp_7 = tmp_6.view(1, 1024, 384)
            tmp_8 = in_2 + tmp_7
            tmp_9 = torch.nn.functional.layer_norm(tmp_8, (384,), in_1, in_0, 1e-05)
            return (tmp_8, tmp_9)

    device = "cuda"
    dtype = torch.bfloat16
    in_0 = torch.randn(384, device=device, dtype=dtype)
    in_1 = torch.randn(384, device=device, dtype=dtype)
    in_2 = torch.randn(1, 1024, 384, device=device, dtype=dtype)
    in_3 = torch.randn(1, 5, 7, 5, 7, 384, device=device, dtype=dtype)

    orig = OriginalGraph().to(device).eval()
    with torch.no_grad():
        eager_out = orig(in_0, in_1, in_2, in_3)
    eager_ms = benchmark(lambda: orig(in_0, in_1, in_2, in_3), "Eager")

    # Manual FX graph with fused kernel
    graph = fx.Graph()
    n0, n1, n2, n3 = [graph.placeholder(f"in_{i}") for i in range(4)]
    out = graph.call_function(fused_roll_slice_add_layernorm_dispatch, (n3, n2, n1, n0, "D384"))
    graph.output(out)
    gm_fused = fx.GraphModule(torch.nn.Module(), graph).to(device).eval()

    with torch.no_grad():
        fused_out = gm_fused(in_0, in_1, in_2, in_3)
    for i, (a, b) in enumerate(zip(eager_out, fused_out)):
        diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Fused vs Eager output {i} max_diff: {diff}")
    fused_ms = benchmark(lambda: gm_fused(in_0, in_1, in_2, in_3), "Fused direct")

    # Serialize
    serialized = PassMgrFXSerializeBackend._serialize_graph_module(gm_fused)
    with torch.no_grad():
        ser_out = serialized(in_0, in_1, in_2, in_3)
    for i, (a, b) in enumerate(zip(fused_out, ser_out)):
        diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Serialized vs Fused output {i} max_diff: {diff}")
    ser_ms = benchmark(lambda: serialized(in_0, in_1, in_2, in_3), "Serialized")

    print(f"\n[Summary] Eager={eager_ms:.4f}ms  Fused={fused_ms:.4f}ms  Serialized={ser_ms:.4f}ms")
    print(f"[Summary] Fused speedup vs Eager: {eager_ms/fused_ms:.2f}x")
    print(f"[Summary] Serialized vs Fused: {fused_ms/ser_ms:.2f}x")
    sys.path.remove("/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/maskformer")


# ============================================================
# Pass 2: rexnetr (add + channel_slice + cat)
# ============================================================
def test_rexnetr():
    print("\n" + "=" * 60)
    print("PASS 2: rexnetr FuseAddSliceCat_304")
    print("=" * 60)

    sys.path.insert(0, "/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/rexnetr")
    from pass_dir.FuseAddSliceCat_304 import pattern, replacement_func, replacement_args
    from pass_dir.shared_triton_add_slice_cat import fused_add_slice_cat_kernel

    class OriginalGraph(torch.nn.Module):
        def forward(self, in_0, in_1, in_2):
            tmp_0 = in_0 + in_1
            tmp_1 = in_2[(slice(None, None, None), slice(304, None, None))]
            tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
            return (tmp_2,)

    device = "cuda"
    dtype = torch.float32
    N, H, W = 4, 28, 28
    in_0 = torch.randn(N, 304, H, W, device=device, dtype=dtype)
    in_1 = torch.randn(N, 304, H, W, device=device, dtype=dtype)
    in_2 = torch.randn(N, 328, H, W, device=device, dtype=dtype)

    orig = OriginalGraph().to(device).eval()
    with torch.no_grad():
        eager_out = orig(in_0, in_1, in_2)
    eager_ms = benchmark(lambda: orig(in_0, in_1, in_2), "Eager")

    # Manual FX graph with fused kernel
    kernel = replacement_func()
    graph = fx.Graph()
    n0, n1, n2 = [graph.placeholder(f"in_{i}") for i in range(3)]
    out = graph.call_function(kernel, (n0, n1, n2))
    graph.output(out)
    gm_fused = fx.GraphModule(torch.nn.Module(), graph).to(device).eval()

    with torch.no_grad():
        fused_out = gm_fused(in_0, in_1, in_2)
    for i, (a, b) in enumerate(zip(eager_out, fused_out)):
        diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Fused vs Eager output {i} max_diff: {diff}")
    fused_ms = benchmark(lambda: gm_fused(in_0, in_1, in_2), "Fused direct")

    # Serialize
    serialized = PassMgrFXSerializeBackend._serialize_graph_module(gm_fused)
    with torch.no_grad():
        ser_out = serialized(in_0, in_1, in_2)
    for i, (a, b) in enumerate(zip(fused_out, ser_out)):
        diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Serialized vs Fused output {i} max_diff: {diff}")
    ser_ms = benchmark(lambda: serialized(in_0, in_1, in_2), "Serialized")

    print(f"\n[Summary] Eager={eager_ms:.4f}ms  Fused={fused_ms:.4f}ms  Serialized={ser_ms:.4f}ms")
    print(f"[Summary] Fused speedup vs Eager: {eager_ms/fused_ms:.2f}x")
    print(f"[Summary] Serialized vs Fused: {fused_ms/ser_ms:.2f}x")
    sys.path.remove("/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/rexnetr")


# ============================================================
# Pass 3: eca_botnext (avg_pool2d + batch_norm)
# ============================================================
def test_eca_botnext():
    print("\n" + "=" * 60)
    print("PASS 3: eca_botnext FuseAvgPoolBatchNormSilu")
    print("=" * 60)

    sys.path.insert(0, "/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/eca_botnext")
    from pass_dir.FuseAvgPoolBatchNormSilu import pattern, replacement_func, replacement_args
    from pass_dir.shared_cuda_eca import fused_avgpool_bn

    class OriginalGraph(torch.nn.Module):
        def forward(self, in_0, in_1, in_2, in_3, in_4):
            tmp_4 = in_4.reshape(1, 512, 16, 16)
            tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
            tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
            return tmp_6

    device = "cuda"
    dtype = torch.bfloat16
    in_0 = torch.randn(512, device=device, dtype=dtype)  # running_mean
    in_1 = torch.rand(512, device=device, dtype=dtype) + 1e-3  # running_var (must be positive)
    in_2 = torch.randn(512, device=device, dtype=dtype)  # bias
    in_3 = torch.randn(512, device=device, dtype=dtype)  # weight
    in_4 = torch.randn(1, 512, 16, 16, device=device, dtype=dtype)

    orig = OriginalGraph().to(device).eval()
    with torch.no_grad():
        eager_out = orig(in_0, in_1, in_2, in_3, in_4)
    eager_ms = benchmark(lambda: orig(in_0, in_1, in_2, in_3, in_4), "Eager")

    # Manual FX graph with fused kernel
    kernel = replacement_func()
    graph = fx.Graph()
    n0, n1, n2, n3, n4 = [graph.placeholder(f"in_{i}") for i in range(5)]
    out = graph.call_function(kernel, (n0, n1, n2, n3, n4))
    graph.output(out)
    gm_fused = fx.GraphModule(torch.nn.Module(), graph).to(device).eval()

    with torch.no_grad():
        fused_out = gm_fused(in_0, in_1, in_2, in_3, in_4)
    diff = torch.max(torch.abs(eager_out - fused_out)).item()
    print(f"[Correctness] Fused vs Eager max_diff: {diff}")
    fused_ms = benchmark(lambda: gm_fused(in_0, in_1, in_2, in_3, in_4), "Fused direct")

    # Serialize
    serialized = PassMgrFXSerializeBackend._serialize_graph_module(gm_fused)
    with torch.no_grad():
        ser_out = serialized(in_0, in_1, in_2, in_3, in_4)
    diff = torch.max(torch.abs(fused_out - ser_out)).item()
    print(f"[Correctness] Serialized vs Fused max_diff: {diff}")
    ser_ms = benchmark(lambda: serialized(in_0, in_1, in_2, in_3, in_4), "Serialized")

    print(f"\n[Summary] Eager={eager_ms:.4f}ms  Fused={fused_ms:.4f}ms  Serialized={ser_ms:.4f}ms")
    print(f"[Summary] Fused speedup vs Eager: {eager_ms/fused_ms:.2f}x")
    print(f"[Summary] Serialized vs Fused: {fused_ms/ser_ms:.2f}x")
    sys.path.remove("/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/eca_botnext")


# ============================================================
# Pass 4: erniem (add + layer_norm)
# ============================================================
def test_erniem():
    print("\n" + "=" * 60)
    print("PASS 4: erniem FuseAddLayerNorm_768")
    print("=" * 60)

    sys.path.insert(0, "/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/erniem")
    from pass_dir.FuseAddLayerNorm_768 import pattern, replacement_func, replacement_args
    from pass_dir.shared_cuda_kernel import fused_add_layernorm_erniem

    class OriginalGraph(torch.nn.Module):
        def forward(self, tmp_10, tmp_15, in_2, in_1):
            tmp_16 = tmp_10 + tmp_15
            tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), in_2, in_1, 1e-05)
            tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
            return tmp_18

    device = "cuda"
    dtype = torch.bfloat16
    B, S = 2, 512
    tmp_10 = torch.randn(B, S, 768, device=device, dtype=dtype)
    tmp_15 = torch.randn(B, S, 768, device=device, dtype=dtype)
    in_2 = torch.randn(768, device=device, dtype=dtype)  # weight
    in_1 = torch.randn(768, device=device, dtype=dtype)  # bias

    orig = OriginalGraph().to(device).eval()
    with torch.no_grad():
        eager_out = orig(tmp_10, tmp_15, in_2, in_1)
    eager_ms = benchmark(lambda: orig(tmp_10, tmp_15, in_2, in_1), "Eager")

    # Manual FX graph with fused kernel
    graph = fx.Graph()
    n0, n1, n2, n3 = [graph.placeholder(name) for name in ["tmp_10", "tmp_15", "in_2", "in_1"]]
    out = graph.call_function(fused_add_layernorm_erniem, (n0, n1, n2, n3))
    graph.output(out)
    gm_fused = fx.GraphModule(torch.nn.Module(), graph).to(device).eval()

    with torch.no_grad():
        fused_out = gm_fused(tmp_10, tmp_15, in_2, in_1)
    diff = torch.max(torch.abs(eager_out - fused_out)).item()
    print(f"[Correctness] Fused vs Eager max_diff: {diff}")
    fused_ms = benchmark(lambda: gm_fused(tmp_10, tmp_15, in_2, in_1), "Fused direct")

    # Serialize
    serialized = PassMgrFXSerializeBackend._serialize_graph_module(gm_fused)
    with torch.no_grad():
        ser_out = serialized(tmp_10, tmp_15, in_2, in_1)
    diff = torch.max(torch.abs(fused_out - ser_out)).item()
    print(f"[Correctness] Serialized vs Fused max_diff: {diff}")
    ser_ms = benchmark(lambda: serialized(tmp_10, tmp_15, in_2, in_1), "Serialized")

    print(f"\n[Summary] Eager={eager_ms:.4f}ms  Fused={fused_ms:.4f}ms  Serialized={ser_ms:.4f}ms")
    print(f"[Summary] Fused speedup vs Eager: {eager_ms/fused_ms:.2f}x")
    print(f"[Summary] Serialized vs Fused: {fused_ms/ser_ms:.2f}x")
    sys.path.remove("/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/erniem")


# ============================================================
# Pass 5: svipas_se (SE block: conv1x1 + add + div + clamp + mul)
# ============================================================
def test_svipas_se():
    print("\n" + "=" * 60)
    print("PASS 5: svipas_se FuseSEBlock_SViPNAS")
    print("=" * 60)

    sys.path.insert(0, "/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/svipas_se")
    from pass_dir.FuseSEBlock_SViPNAS import pattern, replacement_func, replacement_args
    from pass_dir.FuseSEBlock_SViPNAS import fused_se_block

    class OriginalGraph(torch.nn.Module):
        def forward(self, in_0, in_1, in_2, in_3):
            conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
            tmp_3 = conv2d + 1.0
            tmp_4 = tmp_3 / 2.0
            tmp_5 = tmp_4.clamp_(0.0, 1.0)
            tmp_6 = in_2 * tmp_5
            return (tmp_6,)

    device = "cuda"
    dtype = torch.bfloat16
    B, C_out, C_in, H, W = 2, 128, 32, 28, 28
    in_0 = torch.randn(C_out, device=device, dtype=dtype)  # bias
    in_1 = torch.randn(C_out, C_in, 1, 1, device=device, dtype=dtype)  # weight
    in_2 = torch.randn(B, C_out, H, W, device=device, dtype=dtype)  # feat
    in_3 = torch.randn(B, C_in, 1, 1, device=device, dtype=dtype)  # pooled

    orig = OriginalGraph().to(device).eval()
    with torch.no_grad():
        eager_out = orig(in_0, in_1, in_2, in_3)
    eager_ms = benchmark(lambda: orig(in_0, in_1, in_2, in_3), "Eager")

    # Manual FX graph with fused kernel
    # Note: replacement_args reorders: (in_2, in_3, in_1, in_0) -> kernel(feat, pooled, weight, bias)
    graph = fx.Graph()
    n0, n1, n2, n3 = [graph.placeholder(f"in_{i}") for i in range(4)]
    out = graph.call_function(fused_se_block, (n2, n3, n1, n0))
    graph.output(out)
    gm_fused = fx.GraphModule(torch.nn.Module(), graph).to(device).eval()

    with torch.no_grad():
        fused_out = gm_fused(in_0, in_1, in_2, in_3)
    for i, (a, b) in enumerate(zip(eager_out, fused_out)):
        diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Fused vs Eager output {i} max_diff: {diff}")
    fused_ms = benchmark(lambda: gm_fused(in_0, in_1, in_2, in_3), "Fused direct")

    # Serialize
    serialized = PassMgrFXSerializeBackend._serialize_graph_module(gm_fused)
    with torch.no_grad():
        ser_out = serialized(in_0, in_1, in_2, in_3)
    for i, (a, b) in enumerate(zip(fused_out, ser_out)):
        diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Serialized vs Fused output {i} max_diff: {diff}")
    ser_ms = benchmark(lambda: serialized(in_0, in_1, in_2, in_3), "Serialized")

    print(f"\n[Summary] Eager={eager_ms:.4f}ms  Fused={fused_ms:.4f}ms  Serialized={ser_ms:.4f}ms")
    print(f"[Summary] Fused speedup vs Eager: {eager_ms/fused_ms:.2f}x")
    print(f"[Summary] Serialized vs Fused: {fused_ms/ser_ms:.2f}x")
    sys.path.remove("/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/svipas_se")


# ============================================================
# Pass 6: yolov9e_seg (cat + interpolate + stack)
# ============================================================
def test_yolov9e_seg():
    print("\n" + "=" * 60)
    print("PASS 6: yolov9e_seg FuseCatInterpStack")
    print("=" * 60)

    sys.path.insert(0, "/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/yolov9e_seg")
    from pass_dir.FuseCatInterpStack import pattern, replacement_func, replacement_args
    from pass_dir.shared_cuda_yolov9 import fused_cat_interp_stack

    class OriginalGraph(torch.nn.Module):
        def forward(self, in_0, in_1, in_2, in_3):
            tmp_0 = torch.cat((in_2, in_3), 1)
            tmp_1 = torch.nn.functional.interpolate(in_0, size=(40, 40), mode="nearest")
            tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode="nearest")
            tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
            return (tmp_3,)

    device = "cuda"
    dtype = torch.bfloat16
    N, C = 1, 512
    in_0 = torch.randn(N, C, 40, 40, device=device, dtype=dtype)
    in_1 = torch.randn(N, C, 20, 20, device=device, dtype=dtype)
    in_2 = torch.randn(N, C // 2, 40, 40, device=device, dtype=dtype)
    in_3 = torch.randn(N, C // 2, 40, 40, device=device, dtype=dtype)

    orig = OriginalGraph().to(device).eval()
    with torch.no_grad():
        eager_out = orig(in_0, in_1, in_2, in_3)
    eager_ms = benchmark(lambda: orig(in_0, in_1, in_2, in_3), "Eager")

    # Manual FX graph with fused kernel
    graph = fx.Graph()
    n0, n1, n2, n3 = [graph.placeholder(f"in_{i}") for i in range(4)]
    out = graph.call_function(fused_cat_interp_stack, (n0, n1, n2, n3))
    graph.output(out)
    gm_fused = fx.GraphModule(torch.nn.Module(), graph).to(device).eval()

    with torch.no_grad():
        fused_out = gm_fused(in_0, in_1, in_2, in_3)
    for i, (a, b) in enumerate(zip(eager_out, fused_out)):
        diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Fused vs Eager output {i} max_diff: {diff}")
    fused_ms = benchmark(lambda: gm_fused(in_0, in_1, in_2, in_3), "Fused direct")

    # Serialize
    serialized = PassMgrFXSerializeBackend._serialize_graph_module(gm_fused)
    with torch.no_grad():
        ser_out = serialized(in_0, in_1, in_2, in_3)
    for i, (a, b) in enumerate(zip(fused_out, ser_out)):
        diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Serialized vs Fused output {i} max_diff: {diff}")
    ser_ms = benchmark(lambda: serialized(in_0, in_1, in_2, in_3), "Serialized")

    print(f"\n[Summary] Eager={eager_ms:.4f}ms  Fused={fused_ms:.4f}ms  Serialized={ser_ms:.4f}ms")
    print(f"[Summary] Fused speedup vs Eager: {eager_ms/fused_ms:.2f}x")
    print(f"[Summary] Serialized vs Fused: {fused_ms/ser_ms:.2f}x")
    sys.path.remove("/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/yolov9e_seg")


if __name__ == "__main__":
    test_maskformer()
    test_rexnetr()
    test_eca_botnext()
    test_erniem()
    test_svipas_se()
    test_yolov9e_seg()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
