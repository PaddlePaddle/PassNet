import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = w_4
        tmp_5 = w_5
        tmp_6 = torch.nn.functional.scaled_dot_product_attention(in_3, in_2, in_4, attn_mask=in_0, dropout_p=0.0, is_causal=False)
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = tmp_7.contiguous()
        tmp_7 = None
        tmp_9 = tmp_8.view(2, -1, 768)
        tmp_8 = None
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_1, tmp_0)
        tmp_9 = tmp_1 = tmp_0 = None
        tmp_11 = tmp_10 + in_1
        tmp_10 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (768,), tmp_5, tmp_4, 1e-12)
        tmp_11 = tmp_5 = tmp_4 = None
        tmp_13 = torch.nn.functional.linear(tmp_12, tmp_3, tmp_2)
        tmp_3 = tmp_2 = None
        tmp_14 = torch.nn.functional.gelu(tmp_13)
        tmp_13 = None
        return (tmp_12, tmp_14)