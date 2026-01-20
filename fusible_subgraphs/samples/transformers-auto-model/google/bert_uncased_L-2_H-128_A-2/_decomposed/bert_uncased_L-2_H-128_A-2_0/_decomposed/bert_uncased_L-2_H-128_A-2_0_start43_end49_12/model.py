import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, w_0, w_1):
        tmp_0 = torch.nn.functional.linear(in_1, w_1, w_0)
        tmp_1 = tmp_0.view(1, -1, 2, 64)
        tmp_0 = None
        tmp_2 = tmp_1.transpose(1, 2)
        tmp_1 = None
        tmp_3 = torch.nn.functional.scaled_dot_product_attention(in_2, in_3, tmp_2, attn_mask=in_0, dropout_p=0.0, is_causal=False)
        tmp_2 = None
        tmp_4 = tmp_3.transpose(1, 2)
        tmp_3 = None
        tmp_5 = tmp_4.reshape(1, 12, 128)
        tmp_4 = None
        return (tmp_5,)