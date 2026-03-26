import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14):
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_13, in_12, in_14, attn_mask = in_10, dropout_p = 0.0, is_causal = False);  in_13 = in_12 = in_14 = in_10 = None
        tmp_11 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_12 = tmp_11.contiguous();  tmp_11 = None
        tmp_13 = tmp_12.view(128, -1, 768);  tmp_12 = None
        linear = torch.nn.functional.linear(tmp_13, in_1, in_0);  tmp_13 = in_1 = in_0 = None
        tmp_15 = linear + in_11;  linear = in_11 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (768,), in_9, in_8, 1e-12);  tmp_15 = in_9 = in_8 = None
        linear_1 = torch.nn.functional.linear(tmp_16, in_3, in_2);  in_3 = in_2 = None
        tmp_18 = torch.nn.functional.gelu(linear_1);  linear_1 = None
        linear_2 = torch.nn.functional.linear(tmp_18, in_5, in_4);  tmp_18 = in_5 = in_4 = None
        tmp_20 = torch.nn.functional.dropout(linear_2, 0.1, False, False);  linear_2 = None
        tmp_21 = tmp_20 + tmp_16;  tmp_20 = tmp_16 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (768,), in_7, in_6, 1e-12);  tmp_21 = in_7 = in_6 = None
        return (tmp_22,)
        