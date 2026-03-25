import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_2, in_1, in_3, in_0, 0.0, is_causal = False);  in_2 = in_1 = in_3 = in_0 = None
        return (scaled_dot_product_attention,)
        