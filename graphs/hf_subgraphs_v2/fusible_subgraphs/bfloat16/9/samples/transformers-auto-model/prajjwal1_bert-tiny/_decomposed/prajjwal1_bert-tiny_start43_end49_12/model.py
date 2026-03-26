import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear.view(1, -1, 2, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_5, in_4, tmp_4, attn_mask = in_2, dropout_p = 0.0, is_causal = False);  in_5 = in_4 = tmp_4 = in_2 = None
        tmp_6 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_7 = tmp_6.reshape(1, 11, 128);  tmp_6 = None
        return (tmp_7,)
        