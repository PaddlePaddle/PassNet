import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = torch.nn.functional.scaled_dot_product_attention(in_9, in_8, in_10, attn_mask=in_6, dropout_p=0.0, is_causal=False)
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = tmp_7.contiguous()
        tmp_7 = None
        tmp_9 = tmp_8.view(8, -1, 32)
        tmp_8 = None
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_1, tmp_0)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_11 = tmp_10 + in_7
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (32,), tmp_5, tmp_4, 1e-12)
        tmp_11 = tmp_5 = tmp_4 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_14 = torch.nn.functional.gelu(tmp_13)
        tmp_13 = None
        return (tmp_12, tmp_14)