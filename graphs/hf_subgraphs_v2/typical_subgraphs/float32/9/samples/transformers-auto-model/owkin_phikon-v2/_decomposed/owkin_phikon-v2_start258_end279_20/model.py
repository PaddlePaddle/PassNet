import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1, in_2, in_3):
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
        tmp_10 = w_10
        tmp_11 = w_11
        tmp_12 = w_12
        tmp_13 = w_13
        tmp_14 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_15 = tmp_14.view(1, -1, 16, 64)
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = tmp_16.contiguous()
        tmp_16 = None
        tmp_18 = in_0.contiguous()
        tmp_19 = in_3.contiguous()
        tmp_20 = torch.nn.functional.scaled_dot_product_attention(tmp_17, tmp_18, tmp_19, attn_mask=None, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_17 = tmp_18 = tmp_19 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_21.contiguous()
        tmp_21 = None
        tmp_23 = tmp_22.reshape((1, 197, 1024))
        tmp_22 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_5, tmp_4)
        tmp_23 = tmp_5 = tmp_4 = None
        tmp_25 = torch.nn.functional.dropout(tmp_24, 0.0, False, False)
        tmp_24 = None
        tmp_26 = tmp_25 * tmp_6
        tmp_25 = tmp_6 = None
        tmp_27 = tmp_26 + in_2
        tmp_26 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (1024,), tmp_13, tmp_12, 1e-06)
        tmp_13 = tmp_12 = None
        tmp_29 = torch.nn.functional.linear(tmp_28, tmp_9, tmp_8)
        tmp_28 = tmp_9 = tmp_8 = None
        tmp_30 = torch.nn.functional.gelu(tmp_29)
        tmp_29 = None
        tmp_31 = torch.nn.functional.linear(tmp_30, tmp_11, tmp_10)
        tmp_30 = tmp_11 = tmp_10 = None
        tmp_32 = tmp_31 * tmp_7
        tmp_31 = tmp_7 = None
        tmp_33 = tmp_32 + tmp_27
        tmp_32 = tmp_27 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (1024,), tmp_1, tmp_0, 1e-06)
        tmp_1 = tmp_0 = None
        return (tmp_34, tmp_33)