import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor, in_8 : torch.Tensor):
        conv2d = torch.conv2d(in_8, in_5, in_4, (16, 16), (0, 0), (1, 1), 1);  in_8 = in_5 = in_4 = None
        tmp_10 = conv2d.flatten(2);  conv2d = None
        tmp_11 = tmp_10.transpose(1, 2);  tmp_10 = None
        tmp_12 = in_6.expand(1, -1, -1);  in_6 = None
        tmp_13 = torch.cat([tmp_12, tmp_11], dim = 1);  tmp_12 = tmp_11 = None
        tmp_14 = tmp_13 + in_7;  tmp_13 = in_7 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (1024,), in_3, in_2, 1e-06);  in_3 = in_2 = None
        linear = torch.nn.functional.linear(tmp_16, in_1, in_0);  tmp_16 = in_1 = in_0 = None
        return (linear, tmp_15)
        