import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10):
        tmp_9 = torch.nn.functional.gelu(in_10);  in_10 = None
        linear = torch.nn.functional.linear(tmp_9, in_8, in_7);  tmp_9 = in_8 = in_7 = None
        tmp_11 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_12 = tmp_11 + in_9;  tmp_11 = in_9 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1024,), in_6, in_5, 1e-12);  in_6 = in_5 = None
        tmp_14 = torch.zeros_like(in_4, requires_grad = False)
        linear_1 = torch.nn.functional.linear(input = tmp_13, weight = in_0, bias = tmp_14);  in_0 = tmp_14 = None
        linear_2 = torch.nn.functional.linear(input = tmp_13, weight = in_2, bias = in_4);  in_2 = in_4 = None
        linear_3 = torch.nn.functional.linear(input = tmp_13, weight = in_1, bias = in_3);  tmp_13 = in_1 = in_3 = None
        tmp_18 = linear_1.view(1, -1, 16, 64);  linear_1 = None
        tmp_19 = tmp_18.transpose(1, 2);  tmp_18 = None
        tmp_20 = linear_2.view(1, -1, 16, 64);  linear_2 = None
        tmp_21 = tmp_20.transpose(1, 2);  tmp_20 = None
        tmp_22 = linear_3.view(1, -1, 16, 64);  linear_3 = None
        tmp_23 = tmp_22.transpose(1, 2);  tmp_22 = None
        tmp_24 = tmp_23.contiguous();  tmp_23 = None
        tmp_25 = tmp_19.contiguous();  tmp_19 = None
        tmp_26 = tmp_21.contiguous();  tmp_21 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_24, tmp_25, tmp_26, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_24 = tmp_25 = tmp_26 = None
        tmp_28 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_29 = tmp_28.contiguous();  tmp_28 = None
        tmp_30 = tmp_29.reshape((1, 1568, 1024));  tmp_29 = None
        return (tmp_30, tmp_12)
        