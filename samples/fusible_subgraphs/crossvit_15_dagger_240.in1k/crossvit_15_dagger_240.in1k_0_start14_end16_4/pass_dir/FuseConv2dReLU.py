import torch
import triton
import triton.language as tl

def pattern(x, w):
    # Example: matching a linear followed by relu
    return torch.nn.functional.relu(torch.nn.functional.linear(x, w))

def replacement_args(x, w):
    # Return args needed for the new operator
    return (x, w)

@triton.jit
def linear_relu_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # x: [M, K], w: [N, K]
    # We compute x @ w.T
    # a_ptrs points to x blocks
    a_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # b_ptrs points to w blocks. 
    # We want B[k, n] to correspond to w[n, k].
    # So we map dim 0 of B block (k) to dim 1 of w (k), and dim 1 of B block (n) to dim 0 of w (n).
    b_ptrs = w_ptr + (offs_bn[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_xk
        b_ptrs += BLOCK_K * stride_wk

    # ReLU
    accumulator = tl.maximum(accumulator, 0.0)
    
    # Store
    c = accumulator.to(x_ptr.dtype.element_ty)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = out_ptr + stride_outm * offs_cm[:, None] + stride_outn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@torch.fx.wrap
def linear_relu_call(x, w):
    # Handle arbitrary batch dimensions by flattening
    x_in = x
    if x.ndim > 2:
        x_in = x.reshape(-1, x.shape[-1])
    
    # Ensure contiguous for simple stride handling if needed
    if not x_in.is_contiguous():
        x_in = x_in.contiguous()
        
    M, K = x_in.shape
    N, Kw = w.shape
    
    # Output shape matches input batch structure
    out_shape = list(x.shape[:-1]) + [N]
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    out_2d = out.view(-1, N)
    
    # Heuristics for block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    
    linear_relu_kernel[grid](
        x_in, w, out_2d,
        M, N, K,
        x_in.stride(0), x_in.stride(1),
        w.stride(0), w.stride(1),
        out_2d.stride(0), out_2d.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return out

def replacement_func():
    return linear_relu_call