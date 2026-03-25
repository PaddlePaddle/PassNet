import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor, w_2 : torch.Tensor, w_3 : torch.Tensor, in_1 : torch.Tensor):
        tmp_5 = in_1.view(1, 10, -1, 64);  in_1 = None
        tmp_6 = tmp_5.transpose(1, 2);  tmp_5 = None
        linear = torch.nn.functional.linear(in_0, w_1, w_0);  w_1 = w_0 = None
        linear_1 = torch.nn.functional.linear(in_0, w_3, w_2);  in_0 = w_3 = w_2 = None
        tmp_9 = linear.view(1, 10, -1, 64);  linear = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = linear_1.view(1, 10, -1, 64);  linear_1 = None
        tmp_12 = tmp_11.transpose(1, 2);  tmp_11 = None
        tmp_13 = tmp_6.contiguous();  tmp_6 = None
        tmp_14 = tmp_10.contiguous()
        tmp_15 = tmp_12.contiguous()
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_13, tmp_14, tmp_15, attn_mask = None, dropout_p = 0.0, scale = 0.125, is_causal = False);  tmp_13 = tmp_14 = tmp_15 = None
        tmp_17 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_18 = tmp_17.contiguous();  tmp_17 = None
        tmp_19 = tmp_18.reshape(1, 10, -1);  tmp_18 = None
        tmp_20 = tmp_19.contiguous();  tmp_19 = None
        return (tmp_20, tmp_10, tmp_12)
        