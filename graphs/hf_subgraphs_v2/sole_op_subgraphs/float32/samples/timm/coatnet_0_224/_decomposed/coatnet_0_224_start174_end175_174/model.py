import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.scaled_dot_product_attention(in_3, in_1, in_2, attn_mask=in_0, dropout_p=0.0)
        return (tmp_0,)