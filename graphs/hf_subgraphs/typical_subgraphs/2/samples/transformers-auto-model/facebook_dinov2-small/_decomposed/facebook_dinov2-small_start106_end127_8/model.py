import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = in_9
        tmp_10 = in_10
        tmp_11 = in_11
        tmp_12 = in_12
        tmp_13 = in_13
        tmp_14 = torch.nn.functional.linear(in_15, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_15 = tmp_14.view(1, -1, 6, 64)
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = tmp_16.contiguous()
        tmp_16 = None
        tmp_18 = in_14.contiguous()
        tmp_19 = in_17.contiguous()
        tmp_20 = torch.nn.functional.scaled_dot_product_attention(tmp_17, tmp_18, tmp_19, attn_mask=None, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_17 = tmp_18 = tmp_19 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_21.contiguous()
        tmp_21 = None
        tmp_23 = tmp_22.reshape((1, 257, 384))
        tmp_22 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_3, tmp_2)
        tmp_23 = tmp_3 = tmp_2 = None
        tmp_25 = torch.nn.functional.dropout(tmp_24, 0.0, False, False)
        tmp_24 = None
        tmp_26 = tmp_25 * tmp_4
        tmp_25 = tmp_4 = None
        tmp_27 = tmp_26 + in_16
        tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (384,), tmp_11, tmp_10, 1e-06)
        tmp_11 = tmp_10 = None
        tmp_29 = torch.nn.functional.linear(tmp_28, tmp_7, tmp_6)
        tmp_28 = tmp_7 = tmp_6 = None
        tmp_30 = torch.nn.functional.gelu(tmp_29)
        tmp_29 = None
        tmp_31 = torch.nn.functional.linear(tmp_30, tmp_9, tmp_8)
        tmp_30 = tmp_9 = tmp_8 = None
        tmp_32 = tmp_31 * tmp_5
        tmp_31 = tmp_5 = None
        tmp_33 = tmp_32 + tmp_27
        tmp_32 = tmp_27 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (384,), tmp_13, tmp_12, 1e-06)
        tmp_13 = tmp_12 = None
        return (tmp_34, tmp_33)