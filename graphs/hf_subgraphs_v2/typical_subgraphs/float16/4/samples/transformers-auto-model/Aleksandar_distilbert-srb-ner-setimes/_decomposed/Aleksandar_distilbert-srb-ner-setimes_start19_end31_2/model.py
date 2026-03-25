import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
        linear = torch.nn.functional.linear(in_9, in_3, in_2);  in_3 = in_2 = None
        tmp_9 = linear.view(8, -1, 12, 64);  linear = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_11, in_10, tmp_10, attn_mask = in_8, dropout_p = 0.0, is_causal = False);  in_11 = in_10 = tmp_10 = in_8 = None
        tmp_12 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_13 = tmp_12.contiguous();  tmp_12 = None
        tmp_14 = tmp_13.view(8, -1, 768);  tmp_13 = None
        linear_1 = torch.nn.functional.linear(tmp_14, in_1, in_0);  tmp_14 = in_1 = in_0 = None
        tmp_16 = linear_1 + in_9;  linear_1 = in_9 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), in_7, in_6, 1e-12);  tmp_16 = in_7 = in_6 = None
        linear_2 = torch.nn.functional.linear(tmp_17, in_5, in_4);  in_5 = in_4 = None
        tmp_19 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        return (tmp_17, tmp_19)
        