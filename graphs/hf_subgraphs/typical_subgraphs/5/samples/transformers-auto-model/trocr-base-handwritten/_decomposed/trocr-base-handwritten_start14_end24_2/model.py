import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.linear(in_2, tmp_0, None)
        tmp_0 = None
        tmp_2 = tmp_1.view(1, -1, 12, 64)
        tmp_1 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = tmp_3.contiguous()
        tmp_3 = None
        tmp_5 = in_1.contiguous()
        tmp_6 = in_3.contiguous()
        tmp_7 = torch.nn.functional.scaled_dot_product_attention(tmp_4, tmp_5, tmp_6, attn_mask=None, dropout_p=0.0, scale=0.125, is_causal=False)
        tmp_4 = tmp_5 = tmp_6 = None
        tmp_8 = tmp_7.transpose(1, 2)
        tmp_7 = None
        tmp_9 = tmp_8.contiguous()
        tmp_8 = None
        tmp_10 = tmp_9.reshape((1, 577, 768))
        tmp_9 = None
        return (tmp_10,)