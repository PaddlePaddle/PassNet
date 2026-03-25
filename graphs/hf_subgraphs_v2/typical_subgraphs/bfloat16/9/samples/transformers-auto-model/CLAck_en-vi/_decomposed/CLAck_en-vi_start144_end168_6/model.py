import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, in_1, in_2, in_3):
        tmp_11 = in_3.view(1, 19, -1, 64);  in_3 = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        linear = torch.nn.functional.linear(in_0, w_3, w_2);  w_3 = w_2 = None
        linear_1 = torch.nn.functional.linear(in_0, w_7, w_6);  in_0 = w_7 = w_6 = None
        tmp_15 = linear.view(1, 19, -1, 64);  linear = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = linear_1.view(1, 19, -1, 64);  linear_1 = None
        tmp_18 = tmp_17.transpose(1, 2);  tmp_17 = None
        tmp_19 = in_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 19, None))];  in_1 = None
        tmp_20 = tmp_12.contiguous();  tmp_12 = None
        tmp_21 = tmp_16.contiguous()
        tmp_22 = tmp_18.contiguous()
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_20, tmp_21, tmp_22, attn_mask = tmp_19, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_20 = tmp_21 = tmp_22 = tmp_19 = None
        tmp_24 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_25 = tmp_24.contiguous();  tmp_24 = None
        tmp_26 = tmp_25.reshape(1, 19, -1);  tmp_25 = None
        tmp_27 = tmp_26.contiguous();  tmp_26 = None
        linear_2 = torch.nn.functional.linear(tmp_27, w_5, w_4);  tmp_27 = w_5 = w_4 = None
        tmp_29 = torch.nn.functional.dropout(linear_2, p = 0.1, training = False);  linear_2 = None
        tmp_30 = in_2 + tmp_29;  in_2 = tmp_29 = None
        tmp_31 = torch.nn.functional.layer_norm(tmp_30, (512,), w_1, w_0, 1e-05);  tmp_30 = w_1 = w_0 = None
        linear_3 = torch.nn.functional.linear(tmp_31, w_9, w_8);  w_9 = w_8 = None
        tmp_33 = torch.nn.functional.silu(linear_3, inplace = False);  linear_3 = None
        tmp_34 = torch.nn.functional.dropout(tmp_33, p = 0.0, training = False);  tmp_33 = None
        return (tmp_31, tmp_34, tmp_16, tmp_18)
        