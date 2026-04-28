import torch
import torch.fx as fx
import time
import sys

sys.path.insert(0, "/ssd2/liangtai/ai4c")
sys.path.insert(0, "/ssd2/liangtai/acm_graphnet_anonymous/pass_outputs/maskformer")

from pass_dir.shared_fused_roll_add_ln import fused_roll_slice_add_layernorm_dispatch
from graph_net_bench.torch.backend.pass_mgr_fx_serialize import PassMgrFXSerializeBackend


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


class ReplacedGraph(torch.nn.Module):
    def forward(self, in_0, in_1, in_2, in_3):
        return fused_roll_slice_add_layernorm_dispatch(in_3, in_2, in_1, in_0, "D384")


def benchmark(fn, label, warmup=25, trials=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(trials):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) / trials * 1000
    print(f"[Benchmark] {label}: {avg_ms:.4f} ms")
    return avg_ms


def main():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    in_0 = torch.randn(384, device=device, dtype=dtype)
    in_1 = torch.randn(384, device=device, dtype=dtype)
    in_2 = torch.randn(1, 1024, 384, device=device, dtype=dtype)
    in_3 = torch.randn(1, 5, 7, 5, 7, 384, device=device, dtype=dtype)

    # 1. Eager baseline (original ops)
    orig = OriginalGraph().to(device).eval()
    with torch.no_grad():
        eager_out = orig(in_0, in_1, in_2, in_3)
    eager_ms = benchmark(lambda: orig(in_0, in_1, in_2, in_3), "Eager (original ops)")

    # 2. Fused kernel direct
    fused = ReplacedGraph().to(device).eval()
    with torch.no_grad():
        fused_out = fused(in_0, in_1, in_2, in_3)
    for i, (a, b) in enumerate(zip(eager_out, fused_out)):
        max_diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Fused vs Eager output {i} max_diff: {max_diff}")
    fused_ms = benchmark(lambda: fused(in_0, in_1, in_2, in_3), "Fused CUDA kernel (direct)")

    # 3. FX GraphModule baseline (symbolic trace of original)
    gm_orig = fx.symbolic_trace(OriginalGraph())
    with torch.no_grad():
        gm_orig_out = gm_orig(in_0, in_1, in_2, in_3)
    gm_orig_ms = benchmark(lambda: gm_orig(in_0, in_1, in_2, in_3), "FX GraphModule (original)")

    # 4. FX GraphModule with fused kernel (manual graph construction)
    #    symbolic_trace fails on pybind ext, so build graph by hand
    graph = fx.Graph()
    in_0_node = graph.placeholder("in_0")
    in_1_node = graph.placeholder("in_1")
    in_2_node = graph.placeholder("in_2")
    in_3_node = graph.placeholder("in_3")
    fused_node = graph.call_function(
        fused_roll_slice_add_layernorm_dispatch,
        (in_3_node, in_2_node, in_1_node, in_0_node, "D384"),
    )
    graph.output(fused_node)
    gm_fused = fx.GraphModule(torch.nn.Module(), graph)
    gm_fused.to(device).eval()
    with torch.no_grad():
        gm_fused_out = gm_fused(in_0, in_1, in_2, in_3)
    gm_fused_ms = benchmark(lambda: gm_fused(in_0, in_1, in_2, in_3), "FX GraphModule (fused)")

    # 5. Serialize the fused FX GraphModule
    serialized = PassMgrFXSerializeBackend._serialize_graph_module(gm_fused)
    with torch.no_grad():
        ser_out = serialized(in_0, in_1, in_2, in_3)
    for i, (a, b) in enumerate(zip(fused_out, ser_out)):
        max_diff = torch.max(torch.abs(a - b)).item()
        print(f"[Correctness] Serialized vs Fused output {i} max_diff: {max_diff}")
    ser_ms = benchmark(lambda: serialized(in_0, in_1, in_2, in_3), "SerializedGraphModule (fused)")

    print(f"\n[Summary]")
    print(f"  Eager original:     {eager_ms:.4f} ms")
    print(f"  Fused direct:       {fused_ms:.4f} ms  ({eager_ms/fused_ms:.2f}x vs eager)")
    print(f"  FX GM original:     {gm_orig_ms:.4f} ms")
    print(f"  FX GM fused:        {gm_fused_ms:.4f} ms")
    print(f"  Serialized fused:   {ser_ms:.4f} ms")
    print(f"\n[Overhead Analysis]")
    print(f"  FX overhead vs eager:       {gm_orig_ms/eager_ms:.2f}x")
    print(f"  Serialize overhead vs FX:   {ser_ms/gm_fused_ms:.2f}x")
    print(f"  Serialize vs FX original:   {ser_ms/gm_orig_ms:.2f}x")


if __name__ == "__main__":
    main()
