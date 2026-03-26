import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        linear = torch.nn.functional.linear(in_3, in_2, in_1);  in_3 = in_2 = in_1 = None
        tmp_4 = linear.reshape(1, 49, 8, -1);  linear = None
        split = tmp_4.split([32, 32, 128], dim = 3);  tmp_4 = None
        tmp_6 = split[0]
        tmp_7 = split[1]
        tmp_8 = split[2];  split = None
        tmp_9 = tmp_6.permute(0, 2, 1, 3);  tmp_6 = None
        tmp_10 = tmp_7.permute(0, 2, 1, 3);  tmp_7 = None
        tmp_11 = tmp_8.permute(0, 2, 1, 3);  tmp_8 = None
        tmp_12 = in_0.to(device(type='cuda', index=0));  in_0 = None
        tmp_13 = tmp_10.transpose(-2, -1);  tmp_10 = None
        return (tmp_9, tmp_12, tmp_13, tmp_11)
        