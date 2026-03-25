import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_2, in_1, in_3, attn_mask = in_0, dropout_p = 0.0, is_causal = False);  in_2 = in_1 = in_3 = in_0 = None
        tmp_1 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_2 = tmp_1.reshape(16, 128, 32);  tmp_1 = None
        return (tmp_2,)
        