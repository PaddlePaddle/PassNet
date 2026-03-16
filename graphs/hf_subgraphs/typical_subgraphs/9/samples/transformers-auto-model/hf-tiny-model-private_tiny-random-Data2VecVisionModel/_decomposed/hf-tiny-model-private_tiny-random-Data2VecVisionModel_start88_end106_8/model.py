import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1, in_2, in_3):
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
        tmp_12 = torch.nn.functional.linear(in_1, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_13 = tmp_12.view(1, -1, 4, 8)
        tmp_12 = None
        tmp_14 = tmp_13.transpose(1, 2)
        tmp_13 = None
        tmp_15 = torch.nn.functional.scaled_dot_product_attention(in_3, in_0, tmp_14, attn_mask=None, dropout_p=0.0, is_causal=False, scale=0.35355339059327373)
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 2, 1, 3)
        tmp_15 = None
        tmp_17 = tmp_16.contiguous()
        tmp_16 = None
        tmp_18 = tmp_17.view(1, 226, 32)
        tmp_17 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_3, tmp_2)
        tmp_18 = tmp_3 = tmp_2 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.1, False, False)
        tmp_19 = None
        tmp_21 = tmp_10 * tmp_20
        tmp_10 = tmp_20 = None
        tmp_22 = tmp_21 + in_2
        tmp_21 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (32,), tmp_7, tmp_6, 1e-12)
        tmp_7 = tmp_6 = None
        tmp_24 = torch.nn.functional.linear(tmp_23, tmp_5, tmp_4)
        tmp_23 = tmp_5 = tmp_4 = None
        tmp_25 = torch.nn.functional.gelu(tmp_24)
        tmp_24 = None
        tmp_26 = torch.nn.functional.linear(tmp_25, tmp_9, tmp_8)
        tmp_25 = tmp_9 = tmp_8 = None
        tmp_27 = torch.nn.functional.dropout(tmp_26, 0.1, False, False)
        tmp_26 = None
        tmp_28 = tmp_11 * tmp_27
        tmp_11 = tmp_27 = None
        tmp_29 = tmp_28 + tmp_22
        tmp_28 = tmp_22 = None
        return (tmp_29,)