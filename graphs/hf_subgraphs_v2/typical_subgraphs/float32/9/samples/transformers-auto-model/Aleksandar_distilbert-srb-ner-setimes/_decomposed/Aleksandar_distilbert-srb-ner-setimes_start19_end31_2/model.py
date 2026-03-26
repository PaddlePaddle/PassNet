import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1, in_2, in_3):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = w_6
        tmp_7 = w_7
        tmp_8 = torch.nn.functional.linear(in_1, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_9 = tmp_8.view(1, -1, 12, 64)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = torch.nn.functional.scaled_dot_product_attention(in_3, in_2, tmp_10, attn_mask=in_0, dropout_p=0.0, is_causal=False)
        tmp_10 = None
        tmp_12 = tmp_11.transpose(1, 2)
        tmp_11 = None
        tmp_13 = tmp_12.contiguous()
        tmp_12 = None
        tmp_14 = tmp_13.view(1, -1, 768)
        tmp_13 = None
        tmp_15 = torch.nn.functional.linear(tmp_14, tmp_1, tmp_0)
        tmp_14 = tmp_1 = tmp_0 = None
        tmp_16 = tmp_15 + in_1
        tmp_15 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), tmp_7, tmp_6, 1e-12)
        tmp_16 = tmp_7 = tmp_6 = None
        tmp_18 = torch.nn.functional.linear(tmp_17, tmp_5, tmp_4)
        tmp_5 = tmp_4 = None
        tmp_19 = torch.nn.functional.gelu(tmp_18)
        tmp_18 = None
        return (tmp_17, tmp_19)