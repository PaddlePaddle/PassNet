import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_2, in_0, None);  in_2 = in_0 = None
        tmp_2 = linear.view(1, -1, 12, 64);  linear = None
        tmp_3 = tmp_2.transpose(1, 2);  tmp_2 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        tmp_5 = in_1.contiguous();  in_1 = None
        tmp_6 = in_3.contiguous();  in_3 = None
        to = tmp_4.to(torch.float16);  tmp_4 = None
        to_1 = tmp_5.to(torch.float16);  tmp_5 = None
        to_2 = tmp_6.to(torch.float16);  tmp_6 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(to, to_1, to_2, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  to = to_1 = to_2 = None
        tmp_8 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_9 = tmp_8.contiguous();  tmp_8 = None
        tmp_10 = tmp_9.reshape((1, 577, 768));  tmp_9 = None
        return (tmp_10,)
        