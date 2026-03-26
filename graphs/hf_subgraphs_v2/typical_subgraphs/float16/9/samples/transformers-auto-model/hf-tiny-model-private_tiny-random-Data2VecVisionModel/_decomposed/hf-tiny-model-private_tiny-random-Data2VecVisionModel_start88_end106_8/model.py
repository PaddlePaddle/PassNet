import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10, w_11, in_0, in_1, in_2, in_3):
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  in_1 = w_1 = w_0 = None
        tmp_13 = linear.view(1, -1, 4, 8);  linear = None
        tmp_14 = tmp_13.transpose(1, 2);  tmp_13 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(in_3, in_0, tmp_14, attn_mask = None, dropout_p = 0.0, is_causal = False, scale = 0.35355339059327373);  in_3 = in_0 = tmp_14 = None
        tmp_16 = scaled_dot_product_attention.permute(0, 2, 1, 3);  scaled_dot_product_attention = None
        tmp_17 = tmp_16.contiguous();  tmp_16 = None
        tmp_18 = tmp_17.view(1, 226, 32);  tmp_17 = None
        linear_1 = torch.nn.functional.linear(tmp_18, w_3, w_2);  tmp_18 = w_3 = w_2 = None
        tmp_20 = torch.nn.functional.dropout(linear_1, 0.1, False, False);  linear_1 = None
        tmp_21 = w_10 * tmp_20;  w_10 = tmp_20 = None
        tmp_22 = tmp_21 + in_2;  tmp_21 = in_2 = None
        tmp_23 = torch.nn.functional.layer_norm(tmp_22, (32,), w_7, w_6, 1e-12);  w_7 = w_6 = None
        linear_2 = torch.nn.functional.linear(tmp_23, w_5, w_4);  tmp_23 = w_5 = w_4 = None
        tmp_25 = torch.nn.functional.gelu(linear_2);  linear_2 = None
        linear_3 = torch.nn.functional.linear(tmp_25, w_9, w_8);  tmp_25 = w_9 = w_8 = None
        tmp_27 = torch.nn.functional.dropout(linear_3, 0.1, False, False);  linear_3 = None
        tmp_28 = w_11 * tmp_27;  w_11 = tmp_27 = None
        tmp_29 = tmp_28 + tmp_22;  tmp_28 = tmp_22 = None
        return (tmp_29,)
        