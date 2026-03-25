import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0, w_1, w_2, w_3, in_1, in_2):
        tmp_5 = in_2.view(1, 12, -1, 64);  in_2 = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(in_0, w_3, w_2);  in_0 = w_3 = w_2 = None
        tmp_9 = linear.view(1, 12, -1, 64);  linear = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = linear_1.view(1, 12, -1, 64);  linear_1 = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = in_1[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 12, None))];  in_1 = None
        tmp_14 = tmp_6.contiguous();  tmp_6 = None
        tmp_15 = tmp_10.contiguous()
        tmp_16 = tmp_12.contiguous()
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_14, tmp_15, tmp_16, attn_mask = tmp_13, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_14 = tmp_15 = tmp_16 = tmp_13 = None
        tmp_18 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_19 = tmp_18.contiguous();  tmp_18 = None
        tmp_20 = tmp_19.reshape(1, 12, -1);  tmp_19 = None
        tmp_21 = tmp_20.contiguous();  tmp_20 = None
        return (tmp_21, tmp_10, tmp_12)
        