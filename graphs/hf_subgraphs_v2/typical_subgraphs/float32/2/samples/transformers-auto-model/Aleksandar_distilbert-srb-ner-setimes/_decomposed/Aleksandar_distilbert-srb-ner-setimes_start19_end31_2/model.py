import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = in_7
        tmp_8 = torch.nn.functional.linear(in_9, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_9 = tmp_8.view(16, -1, 12, 64)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = torch.nn.functional.scaled_dot_product_attention(in_11, in_10, tmp_10, attn_mask=in_8, dropout_p=0.0, is_causal=False)
        tmp_10 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = tmp_12.contiguous()
        tmp_12 = None
        tmp_14 = tmp_13.view(16, -1, 768)
        tmp_13 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_1, tmp_0)
        tmp_14 = tmp_1 = tmp_0 = None
        tmp_16 = tmp_15 + in_9
        tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), tmp_7, tmp_6, 1e-12)
        tmp_16 = tmp_7 = tmp_6 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_19 = torch.nn.functional.gelu(tmp_18)
        tmp_18 = None
        return (tmp_17, tmp_19)