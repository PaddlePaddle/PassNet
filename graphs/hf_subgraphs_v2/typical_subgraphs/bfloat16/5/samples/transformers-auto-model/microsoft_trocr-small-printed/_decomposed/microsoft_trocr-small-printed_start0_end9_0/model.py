import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor, in_5 : torch.Tensor, in_6 : torch.Tensor, in_7 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_2, in_1, (16, 16), (0, 0), (1, 1), 1);  in_0 = in_2 = in_1 = None
        tmp_9 = conv2d.flatten(2);  conv2d = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = in_3.expand(1, -1, -1);  in_3 = None
        tmp_12 = in_4.expand(1, -1, -1);  in_4 = None
        tmp_13 = torch.cat((tmp_11, tmp_12, tmp_10), dim = 1);  tmp_11 = tmp_12 = tmp_10 = None
        tmp_14 = tmp_13 + in_5;  tmp_13 = in_5 = None
        tmp_15 = torch.nn.functional.dropout(tmp_14, 0.0, False, False);  tmp_14 = None
        tmp_16 = torch.nn.functional.layer_norm(tmp_15, (384,), in_7, in_6, 1e-12);  in_7 = in_6 = None
        return (tmp_15, tmp_16)
        