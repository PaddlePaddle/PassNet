import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14):
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
        tmp_10 = torch.nn.functional.scaled_dot_product_attention(in_13, in_12, in_14, attn_mask=in_10, dropout_p=0.0, is_causal=False)
        tmp_11 = tmp_10.transpose(1, 2)
        tmp_10 = None
        tmp_12 = tmp_11.contiguous()
        tmp_11 = None
        tmp_13 = tmp_12.view(1, -1, 768)
        tmp_12 = None
        tmp_14 = torch.nn.functional.linear(tmp_13, tmp_1, tmp_0)
        tmp_13 = tmp_1 = tmp_0 = None
        tmp_15 = tmp_14 + in_11
        tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (768,), tmp_9, tmp_8, 1e-12)
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
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (768,), tmp_7, tmp_6, 1e-12)
        tmp_21 = tmp_7 = tmp_6 = None
        return (tmp_22,)