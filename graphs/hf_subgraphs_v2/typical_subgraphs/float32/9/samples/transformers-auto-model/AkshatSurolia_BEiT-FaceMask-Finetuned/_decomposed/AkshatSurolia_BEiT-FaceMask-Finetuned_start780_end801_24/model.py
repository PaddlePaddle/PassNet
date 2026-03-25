import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1, in_2, in_3, in_4):
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
        tmp_14 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_15 = tmp_14.view(1, -1, 12, 64)
        tmp_14 = None
        tmp_16 = tmp_15.transpose(1, 2)
        tmp_15 = None
        tmp_17 = torch.nn.functional.scaled_dot_product_attention(in_3, in_0, tmp_16, attn_mask=in_4, dropout_p=0.0, is_causal=False, scale=0.125)
        tmp_16 = None
        tmp_18 = tmp_17.permute(0, 2, 1, 3)
        tmp_17 = None
        tmp_19 = tmp_18.contiguous()
        tmp_18 = None
        tmp_20 = tmp_19.view(1, 197, 768)
        tmp_19 = None
        tmp_21 = torch.nn.functional.linear(tmp_20, tmp_3, tmp_2)
        tmp_20 = tmp_3 = tmp_2 = None
        tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
        tmp_21 = None
        tmp_23 = tmp_10 * tmp_22
        tmp_10 = tmp_22 = None
        tmp_24 = tmp_23 + in_2
        tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (768,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_5, tmp_4)
        tmp_25 = tmp_5 = tmp_4 = None
        tmp_27 = torch.nn.functional.gelu(tmp_26)
        tmp_26 = None
        tmp_28 = torch.nn.functional.linear(tmp_27, tmp_9, tmp_8)
        tmp_27 = tmp_9 = tmp_8 = None
        tmp_29 = torch.nn.functional.dropout(tmp_28, 0.0, False, False)
        tmp_28 = None
        tmp_30 = tmp_11 * tmp_29
        tmp_11 = tmp_29 = None
        tmp_31 = tmp_30 + tmp_24
        tmp_30 = tmp_24 = None
        tmp_32 = tmp_31[slice(None, None, None), slice(1, None, None), slice(None, None, None)]
        tmp_33 = tmp_32.mean(1)
        tmp_32 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (768,), tmp_13, tmp_12, 1e-12)
        tmp_33 = tmp_13 = tmp_12 = None
        return (tmp_31, tmp_34)