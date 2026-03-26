import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16, in_17):
        linear = torch.nn.functional.linear(in_15, in_3, in_2);  in_15 = in_3 = in_2 = None
        tmp_15 = linear.view(1, -1, 16, 64);  linear = None
        tmp_16 = tmp_15.transpose(1, 2);  tmp_15 = None
        tmp_17 = tmp_16.contiguous();  tmp_16 = None
        tmp_18 = in_14.contiguous();  in_14 = None
        tmp_19 = in_17.contiguous();  in_17 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_17, tmp_18, tmp_19, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_17 = tmp_18 = tmp_19 = None
        tmp_21 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_22 = tmp_21.contiguous();  tmp_21 = None
        tmp_23 = tmp_22.reshape((1, 197, 1024));  tmp_22 = None
        linear_1 = torch.nn.functional.linear(tmp_23, in_5, in_4);  tmp_23 = in_5 = in_4 = None
        tmp_25 = torch.nn.functional.dropout(linear_1, 0.0, False, False);  linear_1 = None
        tmp_26 = tmp_25 * in_6;  tmp_25 = in_6 = None
        tmp_27 = tmp_26 + in_16;  tmp_26 = in_16 = None
        tmp_28 = torch.nn.functional.layer_norm(tmp_27, (1024,), in_13, in_12, 1e-06);  in_13 = in_12 = None
        linear_2 = torch.nn.functional.linear(tmp_28, in_9, in_8);  tmp_28 = in_9 = in_8 = None
        tmp_30 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_30, in_11, in_10);  tmp_30 = in_11 = in_10 = None
        tmp_32 = linear_3 * in_7;  linear_3 = in_7 = None
        tmp_33 = tmp_32 + tmp_27;  tmp_32 = tmp_27 = None
        tmp_34 = torch.nn.functional.layer_norm(tmp_33, (1024,), in_1, in_0, 1e-06);  in_1 = in_0 = None
        return (tmp_34, tmp_33)
        