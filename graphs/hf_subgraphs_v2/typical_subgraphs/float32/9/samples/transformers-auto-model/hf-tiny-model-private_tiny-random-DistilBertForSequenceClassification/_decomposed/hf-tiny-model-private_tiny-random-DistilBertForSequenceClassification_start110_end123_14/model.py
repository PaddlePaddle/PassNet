import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3, in_4):
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
        tmp_10 = torch.nn.functional.scaled_dot_product_attention(in_3, in_2, in_4, attn_mask=in_0, dropout_p=0.0, is_causal=False)
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = tmp_11.contiguous()
        tmp_11 = None
        tmp_13 = tmp_12.view(1, -1, 32)
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_1, tmp_0)
        tmp_13 = tmp_1 = tmp_0 = None
        tmp_15 = tmp_14 + in_1
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (32,), tmp_9, tmp_8, 1e-12)
        tmp_15 = tmp_9 = tmp_8 = None
        tmp_17 = torch.nn.functional.linear(tmp_16, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_18 = torch.nn.functional.gelu(tmp_17)
        tmp_17 = None
        tmp_19 = torch.nn.functional.linear(tmp_18, tmp_5, tmp_4)
        tmp_18 = tmp_5 = tmp_4 = None
        tmp_20 = torch.nn.functional.dropout(tmp_19, 0.1, False, False)
        tmp_19 = None
        tmp_21 = tmp_20 + tmp_16
        tmp_20 = tmp_16 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (32,), tmp_7, tmp_6, 1e-12)
        tmp_21 = tmp_7 = tmp_6 = None
        return (tmp_22,)