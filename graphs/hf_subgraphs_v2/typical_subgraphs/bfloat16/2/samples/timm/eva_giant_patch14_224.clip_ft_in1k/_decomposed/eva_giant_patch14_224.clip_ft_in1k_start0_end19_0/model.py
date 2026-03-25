import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor, in_9 : torch.Tensor, in_10 : torch.Tensor):
        conv2d = torch.conv2d(in_10, in_7, in_6, (14, 14), (0, 0), (1, 1), 1);  in_10 = in_7 = in_6 = None
        tmp_12 = conv2d.flatten(2);  conv2d = None
        tmp_13 = tmp_12.transpose(1, 2);  tmp_12 = None
        tmp_14 = in_8.expand(1, -1, -1);  in_8 = None
        tmp_15 = torch.cat([tmp_14, tmp_13], dim = 1);  tmp_14 = tmp_13 = None
        tmp_16 = tmp_15 + in_9;  tmp_15 = in_9 = None
        tmp_17 = torch.nn.functional.dropout(tmp_16, 0.0, False, False);  tmp_16 = None
        tmp_18 = torch.nn.functional.layer_norm(tmp_17, (1408,), in_5, in_4, 1e-06);  in_5 = in_4 = None
        tmp_19 = torch.cat((in_2, in_0, in_3));  in_2 = in_0 = in_3 = None
        linear = torch.nn.functional.linear(tmp_18, weight = in_1, bias = tmp_19);  tmp_18 = in_1 = tmp_19 = None
        tmp_21 = linear.reshape(1, 257, 3, 16, -1);  linear = None
        tmp_22 = tmp_21.permute(2, 0, 3, 1, 4);  tmp_21 = None
        unbind = tmp_22.unbind(0);  tmp_22 = None
        tmp_24 = unbind[0]
        tmp_25 = unbind[1]
        tmp_26 = unbind[2];  unbind = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_24, tmp_25, tmp_26, attn_mask = None, dropout_p = 0.0);  tmp_24 = tmp_25 = tmp_26 = None
        tmp_28 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_29 = tmp_28.reshape(1, 257, 1408);  tmp_28 = None
        return (tmp_17, tmp_29)
        