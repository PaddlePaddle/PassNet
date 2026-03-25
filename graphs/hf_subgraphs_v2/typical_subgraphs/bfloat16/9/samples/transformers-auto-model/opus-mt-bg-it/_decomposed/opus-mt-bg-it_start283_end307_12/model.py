import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_0, in_1, in_2, in_3):
        tmp_10 = in_3.view(1, 1, -1, 64);  in_3 = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        linear = torch.nn.functional.linear(in_1, w_3, w_2);  w_3 = w_2 = None
        linear_1 = torch.nn.functional.linear(in_1, w_7, w_6);  in_1 = w_7 = w_6 = None
        tmp_14 = linear.view(1, 45, -1, 64);  linear = None
        tmp_15 = tmp_14.transpose(1, 2);  tmp_14 = None
        tmp_16 = linear_1.view(1, 45, -1, 64);  linear_1 = None
        tmp_17 = tmp_16.transpose(1, 2);  tmp_16 = None
        tmp_18 = in_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 45, None))];  in_0 = None
        tmp_19 = tmp_11.contiguous();  tmp_11 = None
        tmp_20 = tmp_15.contiguous();  tmp_15 = None
        tmp_21 = tmp_17.contiguous();  tmp_17 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_19, tmp_20, tmp_21, attn_mask = tmp_18, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_19 = tmp_20 = tmp_21 = tmp_18 = None
        tmp_23 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        tmp_25 = tmp_24.reshape(1, 1, -1);  tmp_24 = None
        tmp_26 = tmp_25.contiguous();  tmp_25 = None
        linear_2 = torch.nn.functional.linear(tmp_26, w_5, w_4);  tmp_26 = w_5 = w_4 = None
        tmp_28 = torch.nn.functional.dropout(linear_2, p = 0.1, training = False);  linear_2 = None
        tmp_29 = in_2 + tmp_28;  in_2 = tmp_28 = None
        tmp_30 = torch.nn.functional.layer_norm(tmp_29, (512,), w_1, w_0, 1e-05);  tmp_29 = w_1 = w_0 = None
        linear_3 = torch.nn.functional.linear(tmp_30, w_9, w_8);  w_9 = w_8 = None
        tmp_32 = torch.nn.functional.silu(linear_3, inplace = False);  linear_3 = None
        tmp_33 = torch.nn.functional.dropout(tmp_32, p = 0.0, training = False);  tmp_32 = None
        return (tmp_30, tmp_33)
        