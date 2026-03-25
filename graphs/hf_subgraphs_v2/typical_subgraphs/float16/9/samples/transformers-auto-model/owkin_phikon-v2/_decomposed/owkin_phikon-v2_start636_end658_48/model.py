import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, w_12, w_13, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_15 = linear.view(1, -1, 16, 64);  linear = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = tmp_16.contiguous();  tmp_16 = None
        tmp_18 = in_0.contiguous();  in_0 = None
        tmp_19 = in_3.contiguous();  in_3 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_17, tmp_18, tmp_19, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_17 = tmp_18 = tmp_19 = None
        tmp_21 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_22 = tmp_21.contiguous();  tmp_21 = None
        tmp_23 = tmp_22.reshape((1, 197, 1024));  tmp_22 = None
        linear_1 = torch.nn.functional.linear(tmp_23, w_3, w_2);  tmp_23 = w_3 = w_2 = None
        tmp_25 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_26 = tmp_25 * w_4;  tmp_25 = w_4 = None
        tmp_27 = tmp_26 + in_2;  tmp_26 = in_2 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (1024,), w_11, w_10, 1e-06);  w_11 = w_10 = None
        linear_2 = torch.nn.functional.linear(tmp_28, w_7, w_6);  tmp_28 = w_7 = w_6 = None
        tmp_30 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_30, w_9, w_8);  tmp_30 = w_9 = w_8 = None
        tmp_32 = linear_3 * w_5;  linear_3 = w_5 = None
        tmp_33 = tmp_32 + tmp_27;  tmp_32 = tmp_27 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (1024,), w_13, w_12, 1e-06);  tmp_33 = w_13 = w_12 = None
        tmp_35 = tmp_34[(slice(None, None, None), 0, slice(None, None, None))]
        return (tmp_34, tmp_35)
        