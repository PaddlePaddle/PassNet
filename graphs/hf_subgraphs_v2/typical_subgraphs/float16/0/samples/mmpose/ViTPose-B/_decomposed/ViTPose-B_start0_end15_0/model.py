import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_6, in_5, (16, 16), (2, 2), (1, 1), 1);  in_0 = in_6 = in_5 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = tmp_10 + in_7;  tmp_10 = in_7 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False);  tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), in_4, in_3, 1e-06);  in_4 = in_3 = None
        linear = torch.nn.functional.linear(tmp_13, in_2, in_1);  tmp_13 = in_2 = in_1 = None
        tmp_15 = linear.reshape(1, 192, 3, 12, 64);  linear = None
        tmp_16 = tmp_15.permute(2, 0, 3, 1, 4);  tmp_15 = None
        tmp_17 = tmp_16[0]
        tmp_18 = tmp_16[1]
        tmp_19 = tmp_16[2];  tmp_16 = None
        scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention(tmp_17, tmp_18, tmp_19, dropout_p = 0.0);  tmp_17 = tmp_18 = tmp_19 = None
        tmp_21 = scaled_dot_product_attention.transpose(1, 2);  scaled_dot_product_attention = None
        tmp_22 = tmp_21.reshape(1, 192, 768);  tmp_21 = None
        return (tmp_22, tmp_12)
        