import torch
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Pattern to match (do NOT modify)
# ----------------------------------------------------------------------
def pattern(in_0, w_1, w_0):
    tmp = torch.conv2d(in_0, w_1, w_0, (1, 1), (0, 0), (1, 1), 1)
    tmp = tmp.view(1, 2, 8, 8)
    out = tmp.sigmoid()
    return out

def replacement_args(in_0, w_1, w_0):
    return (in_0, w_1, w_0)

# ----------------------------------------------------------------------
# Optimized Triton kernel that fuses conv2d + bias + sigmoid
# ----------------------------------------------------------------------
@triton.jit
def _conv_sigmoid_kernel(
    in_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C_in, H, W,
    C_out, KH, KW,
    stride_h, stride_w,
    padding_h, padding_w,
    H_out, W_out,
    BLOCK_C_IN: tl.constexpr,
):
    # --------------------------------------------------------------
    # Program IDs
    # --------------------------------------------------------------
    pid_c_out = tl.program_id(0)   # output channel
    pid_y = tl.program_id(1)       # output y
    pid_x = tl.program_id(2)       # output x

    # Batch dimension is assumed to be 0 (N == 1 in the pattern)
    n = 0
    c_out = pid_c_out
    y = pid_y
    x = pid_x

    # --------------------------------------------------------------
    # Accumulator (fp32)
    # --------------------------------------------------------------
    acc = 0.0

    # --------------------------------------------------------------
    # Loop over input channels (blocked) and kernel spatial dimensions
    # --------------------------------------------------------------
    for c in range(0, C_in, BLOCK_C_IN):
        c_off = c + tl.arange(0, BLOCK_C_IN)
        mask_c = c_off < C_in

        for kh in range(KH):
            for kw in range(KW):
                # Compute input coordinates for this kernel element
                in_y = y * stride_h - padding_h + kh
                in_x = x * stride_w - padding_w + kw

                # Masks for valid spatial positions
                mask_y = (in_y >= 0) & (in_y < H)
                mask_x = (in_x >= 0) & (in_x < W)
                mask_spatial = mask_y & mask_x

                # ------------------------------------------------------
                # Load input tiles
                # ------------------------------------------------------
                in_offset = (
                    n * C_in * H * W
                    + c_off * H * W
                    + in_y * W
                    + in_x
                )
                inp = tl.load(
                    in_ptr + in_offset,
                    mask=mask_c & mask_spatial,
                    other=0.0
                )

                # ------------------------------------------------------
                # Load weight tiles
                # ------------------------------------------------------
                w_offset = (
                    c_out * C_in * KH * KW
                    + c_off * KH * KW
                    + kh * KW
                    + kw
                )
                wgt = tl.load(
                    weight_ptr + w_offset,
                    mask=mask_c,
                    other=0.0
                )

                # ------------------------------------------------------
                # Accumulate
                # ------------------------------------------------------
                acc += tl.sum(inp * wgt, axis=0)

    # --------------------------------------------------------------
    # Add bias and apply sigmoid
    # --------------------------------------------------------------
    bias = tl.load(bias_ptr + c_out)
    acc = acc + bias
    out_val = 1.0 / (1.0 + tl.exp(-acc))

    # --------------------------------------------------------------
    # Store result
    # --------------------------------------------------------------
    out_offset = (
        n * C_out * H_out * W_out
        + c_out * H_out * W_out
        + y * W_out
        + x
    )
    tl.store(out_ptr + out_offset, out_val)


# ----------------------------------------------------------------------
# Wrapper that will be inserted into the FX graph (must be @torch.fx.wrap)
# ----------------------------------------------------------------------
@torch.fx.wrap
def fused_conv2d_sigmoid(in_0, w_1, w_0):
    """
    Fused Conv2d (stride=1, padding=0) + bias + sigmoid.
    The output is reshaped to match the original view (1, 2, 8, 8).
    """
    N, C_in, H, W = in_0.shape
    C_out, _, KH, KW = w_1.shape

    stride_h = stride_w = 1
    padding_h = padding_w = 0

    H_out = (H + 2 * padding_h - KH) // stride_h + 1
    W_out = (W + 2 * padding_w - KW) // stride_w + 1

    out = torch.empty((N, C_out, H_out, W_out), dtype=in_0.dtype, device=in_0.device)

    # launch grid: one thread per output element (C_out, H_out, W_out)
    grid = (C_out, H_out, W_out)

    _conv_sigmoid_kernel[grid](
        in_0,
        w_1,
        w_0,
        out,
        N, C_in, H, W,
        C_out, KH, KW,
        stride_h, stride_w,
        padding_h, padding_w,
        H_out, W_out,
        BLOCK_C_IN=64,          # tuned block size for typical GPUs
    )

    # reshape to the original view shape expected by the model
    out = out.view(1, 2, 8, 8)
    return out


def replacement_func():
    # Return the callable that will replace the matched subgraph
    return fused_conv2d_sigmoid