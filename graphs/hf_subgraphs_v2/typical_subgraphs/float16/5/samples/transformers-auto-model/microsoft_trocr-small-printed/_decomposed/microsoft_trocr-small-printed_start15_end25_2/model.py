import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        linear = torch.nn.functional.linear(in_3, in_1, in_0);  in_3 = in_1 = in_0 = None
        tmp_3 = linear.view(1, -1, 6, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        tmp_5 = tmp_4.contiguous();  tmp_4 = None
        tmp_6 = in_2.contiguous();  in_2 = None
        tmp_7 = in_4.contiguous();  in_4 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_5, tmp_6, tmp_7, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_5 = tmp_6 = tmp_7 = None
        tmp_9 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_10 = tmp_9.contiguous();  tmp_9 = None
        tmp_11 = tmp_10.reshape((1, 578, 384));  tmp_10 = None
        return (tmp_11,)
        