import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_3 = linear.view(1, -1, 4, 8);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_3, in_2, tmp_4, attn_mask = in_0, dropout_p = 0.0, is_causal = False);  in_3 = in_2 = tmp_4 = in_0 = None
        tmp_6 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_7 = tmp_6.reshape(1, 45, 32);  tmp_6 = None
        return (tmp_7,)
        