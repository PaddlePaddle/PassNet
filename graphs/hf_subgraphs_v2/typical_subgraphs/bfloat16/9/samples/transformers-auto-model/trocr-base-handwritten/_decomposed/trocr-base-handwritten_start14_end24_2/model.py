import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_1, w_0, None);  in_1 = w_0 = None
        tmp_2 = linear.view(1, -1, 12, 64);  linear = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = in_0.contiguous();  in_0 = None
        tmp_6 = in_2.contiguous();  in_2 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_4, tmp_5, tmp_6, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_4 = tmp_5 = tmp_6 = None
        tmp_8 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_9 = tmp_8.contiguous();  tmp_8 = None
        tmp_10 = tmp_9.reshape((1, 577, 768));  tmp_9 = None
        return (tmp_10,)
        