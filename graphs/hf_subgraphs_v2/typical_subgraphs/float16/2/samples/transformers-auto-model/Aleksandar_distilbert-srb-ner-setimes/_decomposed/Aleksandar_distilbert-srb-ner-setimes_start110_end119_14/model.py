import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_9, in_8, in_10, attn_mask = in_6, dropout_p = 0.0, is_causal = False);  in_9 = in_8 = in_10 = in_6 = None
        tmp_7 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_8 = tmp_7.contiguous();  tmp_7 = None
        tmp_9 = tmp_8.view(16, -1, 768);  tmp_8 = None
        linear = torch.nn.functional.linear(tmp_9, in_1, in_0);  tmp_9 = in_1 = in_0 = None
        tmp_11 = linear + in_7;  linear = in_7 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (768,), in_5, in_4, 1e-12);  tmp_11 = in_5 = in_4 = None
        linear_1 = torch.nn.functional.linear(tmp_12, in_3, in_2);  in_3 = in_2 = None
        tmp_14 = torch.nn.functional.gelu(linear_1);  linear_1 = None
        return (tmp_12, tmp_14)
        