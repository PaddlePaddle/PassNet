import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_1, in_0, in_2, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  in_1 = in_0 = in_2 = None
        tmp_1 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_2 = tmp_1.contiguous();  tmp_1 = None
        tmp_3 = tmp_2.reshape((1, 198, 192));  tmp_2 = None
        return (tmp_3,)
        