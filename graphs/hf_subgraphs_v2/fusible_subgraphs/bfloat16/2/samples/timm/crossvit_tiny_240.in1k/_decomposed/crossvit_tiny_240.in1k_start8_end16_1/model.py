import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        conv2d = torch.conv2d(in_6, in_3, in_2, (16, 16), (0, 0), (1, 1), 1);  in_6 = in_3 = in_2 = None
        tmp_8 = conv2d.flatten(2);  conv2d = None
        tmp_9 = tmp_8.transpose(1, 2);  tmp_8 = None
        tmp_10 = in_4.expand(1, -1, -1);  in_4 = None
        tmp_11 = torch.cat((tmp_10, tmp_9), dim = 1);  tmp_10 = tmp_9 = None
        tmp_12 = tmp_11 + in_5;  tmp_11 = in_5 = None
        tmp_13 = torch.nn.functional.dropout(tmp_12, 0.0, False, False);  tmp_12 = None
        tmp_14 = torch.nn.functional.layer_norm(in_7, (96,), in_1, in_0, 1e-06);  in_7 = in_1 = in_0 = None
        return (tmp_14, tmp_13)
        