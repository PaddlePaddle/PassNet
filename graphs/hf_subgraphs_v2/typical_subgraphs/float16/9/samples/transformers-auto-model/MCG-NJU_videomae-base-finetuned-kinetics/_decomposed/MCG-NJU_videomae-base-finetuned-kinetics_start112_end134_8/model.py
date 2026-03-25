import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, in_0, in_1):
        tmp_9 = torch.nn.functional.gelu(in_1);  in_1 = None
        linear = torch.nn.functional.linear(tmp_9, w_1, w_0);  tmp_9 = w_1 = w_0 = None
        tmp_11 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_12 = tmp_11 + in_0;  tmp_11 = in_0 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), w_8, w_7, 1e-12);  w_8 = w_7 = None
        tmp_14 = torch.zeros_like(w_6, requires_grad = False)
        linear_1 = torch.nn.functional.linear(input = tmp_13, weight = w_2, bias = tmp_14);  w_2 = tmp_14 = None
        linear_2 = torch.nn.functional.linear(input = tmp_13, weight = w_4, bias = w_6);  w_4 = w_6 = None
        linear_3 = torch.nn.functional.linear(input = tmp_13, weight = w_3, bias = w_5);  tmp_13 = w_3 = w_5 = None
        tmp_18 = linear_1.view(1, -1, 12, 64);  linear_1 = None
        tmp_19 = tmp_18.transpose(1, 2);  tmp_18 = None
        tmp_20 = linear_2.view(1, -1, 12, 64);  linear_2 = None
        tmp_21 = tmp_20.transpose(1, 2);  tmp_20 = None
        tmp_22 = linear_3.view(1, -1, 12, 64);  linear_3 = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        tmp_25 = tmp_19.contiguous();  tmp_19 = None
        tmp_26 = tmp_21.contiguous();  tmp_21 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_24, tmp_25, tmp_26, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_24 = tmp_25 = tmp_26 = None
        tmp_28 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_29 = tmp_28.contiguous();  tmp_28 = None
        tmp_30 = tmp_29.reshape((1, 1568, 768));  tmp_29 = None
        return (tmp_30, tmp_12)
        