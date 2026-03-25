import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_1, in_0, in_2, attn_mask = None, dropout_p = 0.0, is_causal = False, scale = 0.35355339059327373);  in_1 = in_0 = in_2 = None
        return (scaled_dot_product_attention,)
        