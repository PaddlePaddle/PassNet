import torch
import triton
import triton.language as tl

def pattern(in_1):
    s = in_1.sum(dim=3, keepdim=True)
    out = in_1 / s
    return out

def replacement_args(in_1):
    return (in_1,)

# ----------------------------------------------------------------------
# Triton kernel: compute row‑wise sum and normalize in a single pass
# ----------------------------------------------------------------------
@triton.jit
def _norm_sum_kernel(input_ptr, output_ptr,
                     N, C, H, W,
                     BLOCK_SIZE: tl.constexpr):
    """
    For each (n, c, h) row, compute the sum over the W dimension and
    immediately divide each element by that sum.
    """
    # 1‑D grid over (N, C, H)
    pid = tl.program_id(0)
    n = pid // (C * H)
    tmp = pid % (C * H)
    c = tmp // H
    h = tmp % H

    # 2‑D grid over blocks of W
    block_id = tl.program_id(1)
    w_offset = tl.arange(0, BLOCK_SIZE)
    w = w_offset + block_id * BLOCK_SIZE
    mask = w < W

    # linear offset into the 4‑D tensor
    offset = ((n * C + c) * H + h) * W + w
    x = tl.load(input_ptr + offset, mask=mask, other=0.0)

    # reduction across the entire row (all threads in this row)
    row_sum = tl.sum(x, axis=0)   # broadcasted to every thread

    # avoid division by zero (should not happen for positive inputs)
    row_sum = tl.where(row_sum == 0.0, 1.0, row_sum)

    # normalize
    out = x / row_sum
    tl.store(output_ptr + offset, out, mask=mask)

# ----------------------------------------------------------------------
# Wrapper that will be used by the FX pass
# ----------------------------------------------------------------------
@torch.fx.wrap
def _triton_normalize(in_1):
    """
    High‑performance replacement for:
        out = in_1 / in_1.sum(dim=3, keepdim=True)
    """
    if not in_1.is_cuda:
        raise RuntimeError("Triton kernel requires a CUDA tensor")
    N, C, H, W = in_1.shape
    out = torch.empty_like(in_1)

    BLOCK = 1024
    grid = (N * C * H, (W + BLOCK - 1) // BLOCK)
    _norm_sum_kernel[grid](
        in_1,
        out,
        N, C, H, W,
        BLOCK_SIZE=BLOCK,
    )
    return out

def replacement_func():
    # Return the callable that will be inserted into the graph.
    return _triton_normalize