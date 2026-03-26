import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_15 = linear.view(1, -1, 4, 8);  linear = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_3, in_0, tmp_16, attn_mask = None, dropout_p = 0.0, is_causal = False, scale = 0.35355339059327373);  in_3 = in_0 = tmp_16 = None
        tmp_18 = scaled_dot_product_attention.permute(0, 2, 1, 3);  scaled_dot_product_attention = None
        tmp_19 = tmp_18.contiguous();  tmp_18 = None
        tmp_20 = tmp_19.view(1, 226, 32);  tmp_19 = None
        linear_1 = torch.nn.functional.linear(tmp_20, w_3, w_2);  tmp_20 = w_3 = w_2 = None
        tmp_22 = torch.nn.functional.dropout(linear_1, 0.1, False, False);  linear_1 = None
        tmp_23 = w_10 * tmp_22;  w_10 = tmp_22 = None
        tmp_24 = tmp_23 + in_2;  tmp_23 = in_2 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (32,), w_7, w_6, 1e-12);  w_7 = w_6 = None
        linear_2 = torch.nn.functional.linear(tmp_25, w_5, w_4);  tmp_25 = w_5 = w_4 = None
        tmp_27 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_27, w_9, w_8);  tmp_27 = w_9 = w_8 = None
        tmp_29 = torch.nn.functional.dropout(linear_3, 0.1, False, False);  linear_3 = None
        tmp_30 = w_11 * tmp_29;  w_11 = tmp_29 = None
        tmp_31 = tmp_30 + tmp_24;  tmp_30 = tmp_24 = None
        tmp_32 = tmp_31[(slice(None, None, None), slice(1, None, None), slice(None, None, None))]
        tmp_33 = tmp_32.mean(1);  tmp_32 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (32,), w_13, w_12, 1e-12);  tmp_33 = w_13 = w_12 = None
        return (tmp_31, tmp_34)
        