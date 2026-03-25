import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        conv2d = torch.conv2d(in_7, in_4, in_3, (16, 16), (0, 0), (1, 1), 1);  in_7 = in_4 = in_3 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = in_5.expand(1, -1, -1);  in_5 = None
        tmp_12 = torch.cat([tmp_11, tmp_10], dim = 1);  tmp_11 = tmp_10 = None
        tmp_13 = tmp_12 + in_6;  tmp_12 = in_6 = None
        tmp_14 = torch.nn.functional.dropout(tmp_13, 0.0, False, False);  tmp_13 = None
        tmp_15 = torch.nn.functional.layer_norm(tmp_14, (768,), in_2, in_1, 1e-06);  in_2 = in_1 = None
        linear = torch.nn.functional.linear(tmp_15, in_0, None);  tmp_15 = in_0 = None
        return (linear, tmp_14)
        