import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = w_6
        tmp_8 = w_7
        tmp_9 = w_8
        tmp_10 = w_9
        tmp_11 = in_3.view(1, 12, -1, 64)
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = torch.nn.functional.linear(tmp_0, tmp_4, tmp_3)
        tmp_4 = tmp_3 = None
        tmp_14 = torch.nn.functional.linear(tmp_0, tmp_8, tmp_7)
        tmp_0 = tmp_8 = tmp_7 = None
        tmp_15 = tmp_13.view(1, 12, -1, 64)
        tmp_13 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = tmp_14.view(1, 12, -1, 64)
        tmp_14 = None
        tmp_18 = tmp_17.transpose(1, 2)
        tmp_17 = None
        tmp_19 = in_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 12, None)]
        tmp_20 = tmp_12.contiguous()
        tmp_12 = None
        tmp_21 = tmp_16.contiguous()
        tmp_22 = tmp_18.contiguous()
        tmp_23 = torch.nn.functional.scaled_dot_product_attention(tmp_20, tmp_21, tmp_22, attn_mask=tmp_19, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_20 = tmp_21 = tmp_22 = tmp_19 = None
        tmp_24 = tmp_23.transpose(1, 2)
        tmp_23 = None
        tmp_25 = tmp_24.contiguous()
        tmp_24 = None
        tmp_26 = tmp_25.reshape(1, 12, -1)
        tmp_25 = None
        tmp_27 = tmp_26.contiguous()
        tmp_26 = None
        tmp_28 = torch.nn.functional.linear(tmp_27, tmp_6, tmp_5)
        tmp_27 = tmp_6 = tmp_5 = None
        tmp_29 = torch.nn.functional.dropout(tmp_28, p=0.1, training=False)
        tmp_28 = None
        tmp_30 = in_2 + tmp_29
        tmp_29 = None
        tmp_31 = torch.nn.functional.layer_norm(tmp_30, (512,), tmp_2, tmp_1, 1e-05)
        tmp_30 = tmp_2 = tmp_1 = None
        tmp_32 = torch.nn.functional.linear(tmp_31, tmp_10, tmp_9)
        tmp_10 = tmp_9 = None
        tmp_33 = torch.nn.functional.silu(tmp_32, inplace=False)
        tmp_32 = None
        tmp_34 = torch.nn.functional.dropout(tmp_33, p=0.0, training=False)
        tmp_33 = None
        return (tmp_31, tmp_34, tmp_16, tmp_18)