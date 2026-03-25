import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1 : torch.Tensor, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  in_0 = w_1 = w_0 = None
        tmp_3 = linear.view(2, -1, 2, 64);  linear = None
        tmp_4 = tmp_3.transpose(1, 2);  tmp_3 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_3, in_2, tmp_4, attn_mask = in_1, dropout_p = 0.0, is_causal = False);  in_3 = in_2 = tmp_4 = in_1 = None
        tmp_6 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_7 = tmp_6.reshape(2, 34, 128);  tmp_6 = None
        return (tmp_7,)
        