import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = in_8
        tmp_9 = torch.nn.functional.gelu(in_10)
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_8, tmp_7)
        tmp_9 = tmp_8 = tmp_7 = None
        tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
        tmp_10 = None
        tmp_12 = tmp_11 + in_9
        tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), tmp_6, tmp_5, 1e-12)
        tmp_6 = tmp_5 = None
        tmp_14 = torch.zeros_like(tmp_4, requires_grad=False)
        tmp_15 = torch.nn.functional.linear(input=tmp_13, weight=tmp_0, bias=tmp_14)
        tmp_0 = tmp_14 = None
        tmp_16 = torch.nn.functional.linear(input=tmp_13, weight=tmp_2, bias=tmp_4)
        tmp_2 = tmp_4 = None
        tmp_17 = torch.nn.functional.linear(input=tmp_13, weight=tmp_1, bias=tmp_3)
        tmp_13 = tmp_1 = tmp_3 = None
        tmp_18 = tmp_15.view(1, -1, 12, 64)
        tmp_15 = None
        tmp_19 = tmp_18.transpose(1, 2)
        tmp_18 = None
        tmp_20 = tmp_16.view(1, -1, 12, 64)
        tmp_16 = None
        tmp_21 = tmp_20.transpose(1, 2)
        tmp_20 = None
        tmp_22 = tmp_17.view(1, -1, 12, 64)
        tmp_17 = None
        tmp_23 = tmp_22.transpose(1, 2)
        tmp_22 = None
        tmp_24 = tmp_23.contiguous()
        tmp_23 = None
        tmp_25 = tmp_19.contiguous()
        tmp_19 = None
        tmp_26 = tmp_21.contiguous()
        tmp_21 = None
        tmp_27 = torch.nn.functional.scaled_dot_product_attention(tmp_24, tmp_25, tmp_26, attn_mask=None, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_24 = tmp_25 = tmp_26 = None
        tmp_28 = tmp_27.transpose(1, 2)
        tmp_27 = None
        tmp_29 = tmp_28.contiguous()
        tmp_28 = None
        tmp_30 = tmp_29.reshape((1, 1568, 768))
        tmp_29 = None
        return (tmp_30, tmp_12)