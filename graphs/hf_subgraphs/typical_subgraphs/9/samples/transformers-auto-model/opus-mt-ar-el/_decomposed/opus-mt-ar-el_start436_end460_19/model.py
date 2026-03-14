import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = w_8
        tmp_9 = w_9
        tmp_10 = in_3.view(1, 1, -1, 64)
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = torch.nn.functional.linear(in_2, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_13 = torch.nn.functional.linear(in_2, tmp_7, tmp_6)
        tmp_7 = tmp_6 = None
        tmp_14 = tmp_12.view(1, 56, -1, 64)
        tmp_12 = None
        tmp_15 = tmp_14.transpose(1, 2)
        tmp_14 = None
        tmp_16 = tmp_13.view(1, 56, -1, 64)
        tmp_13 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 56, None)]
        tmp_19 = tmp_11.contiguous()
        tmp_11 = None
        tmp_20 = tmp_15.contiguous()
        tmp_15 = None
        tmp_21 = tmp_17.contiguous()
        tmp_17 = None
        tmp_22 = torch.nn.functional.scaled_dot_product_attention(tmp_19, tmp_20, tmp_21, attn_mask=tmp_18, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_19 = tmp_20 = tmp_21 = tmp_18 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = tmp_23.contiguous()
        tmp_23 = None
        tmp_25 = tmp_24.reshape(1, 1, -1)
        tmp_24 = None
        tmp_26 = tmp_25.contiguous()
        tmp_25 = None
        tmp_27 = torch.nn.functional.linear(tmp_26, tmp_5, tmp_4)
        tmp_26 = tmp_5 = tmp_4 = None
        tmp_28 = torch.nn.functional.dropout(tmp_27, p=0.1, training=False)
        tmp_27 = None
        tmp_29 = in_1 + tmp_28
        tmp_28 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (512,), tmp_1, tmp_0, 1e-05)
        tmp_29 = tmp_1 = tmp_0 = None
        tmp_31 = torch.nn.functional.linear(tmp_30, tmp_9, tmp_8)
        tmp_9 = tmp_8 = None
        tmp_32 = torch.nn.functional.silu(tmp_31, inplace=False)
        tmp_31 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, p=0.0, training=False)
        tmp_32 = None
        return (tmp_30, tmp_33)