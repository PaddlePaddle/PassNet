import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  in_0 = w_1 = w_0 = None
        tmp_3 = linear.view(1, 499, -1, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = in_2.contiguous();  in_2 = None
        tmp_6 = in_1.contiguous();  in_1 = None
        tmp_7 = tmp_4.contiguous();  tmp_4 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_5, tmp_6, tmp_7, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_5 = tmp_6 = tmp_7 = None
        tmp_9 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_10 = tmp_9.contiguous();  tmp_9 = None
        tmp_11 = tmp_10.reshape(1, 499, -1);  tmp_10 = None
        tmp_12 = tmp_11.contiguous();  tmp_11 = None
        return (tmp_12,)
        