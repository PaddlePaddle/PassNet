import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, w_0 : torch.Tensor, w_1, w_2, w_3, in_0, in_1, in_2):
        tmp_4 = in_2.view(1, 1, -1, 64);  in_2 = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        linear = torch.nn.functional.linear(in_1, w_1, w_0);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(in_1, w_3, w_2);  in_1 = w_3 = w_2 = None
        tmp_8 = linear.view(1, 33, -1, 64);  linear = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = linear_1.view(1, 33, -1, 64);  linear_1 = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        tmp_12 = in_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 33, None))];  in_0 = None
        tmp_13 = tmp_5.contiguous();  tmp_5 = None
        tmp_14 = tmp_9.contiguous();  tmp_9 = None
        tmp_15 = tmp_11.contiguous();  tmp_11 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_13, tmp_14, tmp_15, attn_mask = tmp_12, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_13 = tmp_14 = tmp_15 = tmp_12 = None
        tmp_17 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_18 = tmp_17.contiguous();  tmp_17 = None
        tmp_19 = tmp_18.reshape(1, 1, -1);  tmp_18 = None
        tmp_20 = tmp_19.contiguous();  tmp_19 = None
        return (tmp_20,)
        