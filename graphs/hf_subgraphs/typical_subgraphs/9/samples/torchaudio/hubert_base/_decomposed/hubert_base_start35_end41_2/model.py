import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, in_0, in_1, in_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
        tmp_1 = tmp_0 = None
        tmp_3 = tmp_2.view(1, 249, 12, 64)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(2, 1)
        tmp_3 = None
        tmp_5 = torch.nn.functional.scaled_dot_product_attention(in_1, in_0, tmp_4, attn_mask=None, dropout_p=0.0, is_causal=False)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = tmp_6.reshape(1, -1, 768)
        tmp_6 = None
        return (tmp_7,)