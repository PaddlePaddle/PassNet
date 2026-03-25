import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_15, w_14);  in_1 = w_15 = w_14 = None
        tmp_18 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_19 = in_3 + tmp_18;  in_3 = tmp_18 = None
        tmp_20 = torch.nn.functional.layer_norm(tmp_19, (1024,), w_13, w_12, 1e-05);  tmp_19 = w_13 = w_12 = None
        linear_1 = torch.nn.functional.linear(tmp_20, w_7, w_6);  w_7 = w_6 = None
        tmp_22 = linear_1.view(1, 18, -1, 64);  linear_1 = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        linear_2 = torch.nn.functional.linear(in_0, w_3, w_2);  w_3 = w_2 = None
        linear_3 = torch.nn.functional.linear(in_0, w_9, w_8);  in_0 = w_9 = w_8 = None
        tmp_26 = linear_2.view(1, 18, -1, 64);  linear_2 = None
        tmp_27 = tmp_26.transpose(1, 2);  tmp_26 = None
        tmp_28 = linear_3.view(1, 18, -1, 64);  linear_3 = None
        tmp_29 = tmp_28.transpose(1, 2);  tmp_28 = None
        tmp_30 = in_2[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 18, None))];  in_2 = None
        tmp_31 = tmp_23.contiguous();  tmp_23 = None
        tmp_32 = tmp_27.contiguous()
        tmp_33 = tmp_29.contiguous()
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_31, tmp_32, tmp_33, attn_mask = tmp_30, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_31 = tmp_32 = tmp_33 = tmp_30 = None
        tmp_35 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_36 = tmp_35.contiguous();  tmp_35 = None
        tmp_37 = tmp_36.reshape(1, 18, -1);  tmp_36 = None
        tmp_38 = tmp_37.contiguous();  tmp_37 = None
        linear_4 = torch.nn.functional.linear(tmp_38, w_5, w_4);  tmp_38 = w_5 = w_4 = None
        tmp_40 = torch.nn.functional.dropout(linear_4, p = 0.1, training = False);  linear_4 = None
        tmp_41 = tmp_20 + tmp_40;  tmp_20 = tmp_40 = None
        tmp_42 = torch.nn.functional.layer_norm(tmp_41, (1024,), w_1, w_0, 1e-05);  tmp_41 = w_1 = w_0 = None
        linear_5 = torch.nn.functional.linear(tmp_42, w_11, w_10);  w_11 = w_10 = None
        return (tmp_42, tmp_27, linear_5, tmp_29)
        