import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, w_14, w_15, in_0, in_1, in_2):
        linear = torch.nn.functional.linear(in_2, w_1, w_0);  in_2 = w_1 = w_0 = None
        tmp_17 = torch.nn.functional.dropout(linear, p = 0.1, training = False);  linear = None
        tmp_18 = in_1 + tmp_17;  in_1 = tmp_17 = None
        tmp_19 = torch.nn.functional.layer_norm(tmp_18, (768,), w_3, w_2, 1e-05);  tmp_18 = w_3 = w_2 = None
        linear_1 = torch.nn.functional.linear(tmp_19, w_13, w_12);  w_13 = w_12 = None
        tmp_21 = linear_1.view(1, 11, -1, 64);  linear_1 = None
        tmp_22 = tmp_21.transpose(1, 2);  tmp_21 = None
        linear_2 = torch.nn.functional.linear(tmp_19, w_9, w_8);  w_9 = w_8 = None
        linear_3 = torch.nn.functional.linear(tmp_19, w_15, w_14);  w_15 = w_14 = None
        tmp_25 = linear_2.view(1, 11, -1, 64);  linear_2 = None
        tmp_26 = tmp_25.transpose(1, 2);  tmp_25 = None
        tmp_27 = linear_3.view(1, 11, -1, 64);  linear_3 = None
        tmp_28 = tmp_27.transpose(1, 2);  tmp_27 = None
        tmp_29 = in_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 11, None))];  in_0 = None
        tmp_30 = tmp_22.contiguous();  tmp_22 = None
        tmp_31 = tmp_26.contiguous()
        tmp_32 = tmp_28.contiguous()
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_30, tmp_31, tmp_32, attn_mask = tmp_29, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_30 = tmp_31 = tmp_32 = tmp_29 = None
        tmp_34 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_35 = tmp_34.contiguous();  tmp_34 = None
        tmp_36 = tmp_35.reshape(1, 11, -1);  tmp_35 = None
        tmp_37 = tmp_36.contiguous();  tmp_36 = None
        linear_4 = torch.nn.functional.linear(tmp_37, w_11, w_10);  tmp_37 = w_11 = w_10 = None
        tmp_39 = torch.nn.functional.dropout(linear_4, p = 0.1, training = False);  linear_4 = None
        tmp_40 = tmp_19 + tmp_39;  tmp_19 = tmp_39 = None
        tmp_41 = torch.nn.functional.layer_norm(tmp_40, (768,), w_7, w_6, 1e-05);  tmp_40 = w_7 = w_6 = None
        linear_5 = torch.nn.functional.linear(tmp_41, w_5, w_4);  w_5 = w_4 = None
        return (tmp_41, tmp_26, linear_5, tmp_28)
        