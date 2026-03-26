import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3, in_4):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_3, in_2, in_4, attn_mask = in_0, dropout_p = 0.0, is_causal = False);  in_3 = in_2 = in_4 = in_0 = None
        tmp_11 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_12 = tmp_11.contiguous();  tmp_11 = None
        tmp_13 = tmp_12.view(1, -1, 32);  tmp_12 = None
        linear = torch.nn.functional.linear(tmp_13, w_1, w_0);  tmp_13 = w_1 = w_0 = None
        tmp_15 = linear + in_1;  linear = in_1 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (32,), w_9, w_8, 1e-12);  tmp_15 = w_9 = w_8 = None
        linear_1 = torch.nn.functional.linear(tmp_16, w_3, w_2);  w_3 = w_2 = None
        tmp_18 = torch.nn.functional.gelu(linear_1);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_18, w_5, w_4);  tmp_18 = w_5 = w_4 = None
        tmp_20 = torch.nn.functional.dropout(linear_2, 0.1, False, False);  linear_2 = None
        tmp_21 = tmp_20 + tmp_16;  tmp_20 = tmp_16 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (32,), w_7, w_6, 1e-12);  tmp_21 = w_7 = w_6 = None
        return (tmp_22,)
        