import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.scaled_dot_product_attention(in_1, in_0, in_2, attn_mask=None, dropout_p=0.0, scale=0.3535533905932738, is_causal=False)
        return (tmp_0,)