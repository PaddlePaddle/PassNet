import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17, in_18):
        linear = torch.nn.functional.linear(in_15, in_1, in_0);  in_15 = in_1 = in_0 = None
        tmp_15 = linear.view(1, -1, 12, 64);  linear = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_17, in_14, tmp_16, attn_mask = in_18, dropout_p = 0.0, is_causal = False, scale = 0.125);  in_17 = in_14 = tmp_16 = in_18 = None
        tmp_18 = scaled_dot_product_attention.permute(0, 2, 1, 3);  scaled_dot_product_attention = None
        tmp_19 = tmp_18.contiguous();  tmp_18 = None
        tmp_20 = tmp_19.view(1, 197, 768);  tmp_19 = None
        linear_1 = torch.nn.functional.linear(tmp_20, in_3, in_2);  tmp_20 = in_3 = in_2 = None
        tmp_22 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_23 = in_10 * tmp_22;  in_10 = tmp_22 = None
        tmp_24 = tmp_23 + in_16;  tmp_23 = in_16 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (768,), in_7, in_6, 1e-12);  in_7 = in_6 = None
        linear_2 = torch.nn.functional.linear(tmp_25, in_5, in_4);  tmp_25 = in_5 = in_4 = None
        tmp_27 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_27, in_9, in_8);  tmp_27 = in_9 = in_8 = None
        tmp_29 = torch.nn.functional.dropout(linear_3, 0.0, False, False);  linear_3 = None
        tmp_30 = in_11 * tmp_29;  in_11 = tmp_29 = None
        tmp_31 = tmp_30 + tmp_24;  tmp_30 = tmp_24 = None
        tmp_32 = tmp_31[(slice(None, None, None), slice(1, None, None), slice(None, None, None))]
        tmp_33 = tmp_32.mean(1);  tmp_32 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (768,), in_13, in_12, 1e-12);  tmp_33 = in_13 = in_12 = None
        return (tmp_31, tmp_34)
        