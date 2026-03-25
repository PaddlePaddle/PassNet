import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        tmp_8 = torch.nn.functional.gelu(in_9, approximate = 'none');  in_9 = None
        tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False);  tmp_8 = None
        linear = torch.nn.functional.linear(tmp_9, in_1, in_0);  tmp_9 = in_1 = in_0 = None
        tmp_11 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_12 = in_8 + tmp_11;  in_8 = tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1408,), in_7, in_6, 1e-06);  in_7 = in_6 = None
        tmp_14 = torch.cat((in_4, in_2, in_5));  in_4 = in_2 = in_5 = None
        linear_1 = torch.nn.functional.linear(tmp_13, weight = in_3, bias = tmp_14);  tmp_13 = in_3 = tmp_14 = None
        tmp_16 = linear_1.reshape(1, 257, 3, 16, -1);  linear_1 = None
        tmp_17 = tmp_16.permute(2, 0, 3, 1, 4);  tmp_16 = None
        unbind = tmp_17.unbind(0);  tmp_17 = None
        tmp_19 = unbind[0]
        tmp_20 = unbind[1]
        tmp_21 = unbind[2];  unbind = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_19, tmp_20, tmp_21, attn_mask = None, dropout_p = 0.0);  tmp_19 = tmp_20 = tmp_21 = None
        tmp_23 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_24 = tmp_23.reshape(1, 257, 1408);  tmp_23 = None
        return (tmp_12, tmp_24)
        