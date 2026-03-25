import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor):
        conv2d = torch.conv2d(in_6, in_1, in_0, (16, 16), (0, 0), (1, 1), 1);  in_6 = in_1 = in_0 = None
        tmp_7 = conv2d.flatten(2);  conv2d = None
        tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
        tmp_9 = in_2.expand(35, -1, -1);  in_2 = None
        tmp_10 = torch.cat((tmp_9, tmp_8), dim = 1);  tmp_9 = tmp_8 = None
        tmp_11 = tmp_10 + in_3;  tmp_10 = in_3 = None
        tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False);  tmp_11 = None
        tmp_13 = torch.nn.functional.layer_norm(tmp_12, (1024,), in_5, in_4, 1e-06);  in_5 = in_4 = None
        return (tmp_12, tmp_13)
        