import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, in_0, in_1, in_2, in_3, in_4):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_3, in_2, in_4, attn_mask = in_0, dropout_p = 0.0, is_causal = False);  in_3 = in_2 = in_4 = in_0 = None
        tmp_7 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_8 = tmp_7.contiguous();  tmp_7 = None
        tmp_9 = tmp_8.view(1, -1, 32);  tmp_8 = None
        linear = torch.nn.functional.linear(tmp_9, w_1, w_0);  tmp_9 = w_1 = w_0 = None
        tmp_11 = linear + in_1;  linear = in_1 = None
        tmp_12 = torch.nn.functional.layer_norm(tmp_11, (32,), w_5, w_4, 1e-12);  tmp_11 = w_5 = w_4 = None
        linear_1 = torch.nn.functional.linear(tmp_12, w_3, w_2);  w_3 = w_2 = None
        tmp_14 = torch.nn.functional.gelu(linear_1);  linear_1 = None
        return (tmp_12, tmp_14)
        