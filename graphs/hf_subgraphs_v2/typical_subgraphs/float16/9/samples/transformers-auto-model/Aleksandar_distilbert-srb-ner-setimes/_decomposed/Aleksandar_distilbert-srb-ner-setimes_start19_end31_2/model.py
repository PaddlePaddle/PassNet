import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_3, w_2);  w_3 = w_2 = None
        tmp_9 = linear.view(1, -1, 12, 64);  linear = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_3, in_2, tmp_10, attn_mask = in_0, dropout_p = 0.0, is_causal = False);  in_3 = in_2 = tmp_10 = in_0 = None
        tmp_12 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_13 = tmp_12.contiguous();  tmp_12 = None
        tmp_14 = tmp_13.view(1, -1, 768);  tmp_13 = None
        linear_1 = torch.nn.functional.linear(tmp_14, w_1, w_0);  tmp_14 = w_1 = w_0 = None
        tmp_16 = linear_1 + in_1;  linear_1 = in_1 = None
        tmp_17 = torch.nn.functional.layer_norm(tmp_16, (768,), w_7, w_6, 1e-12);  tmp_16 = w_7 = w_6 = None
        linear_2 = torch.nn.functional.linear(tmp_17, w_5, w_4);  w_5 = w_4 = None
        tmp_19 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        return (tmp_17, tmp_19)
        