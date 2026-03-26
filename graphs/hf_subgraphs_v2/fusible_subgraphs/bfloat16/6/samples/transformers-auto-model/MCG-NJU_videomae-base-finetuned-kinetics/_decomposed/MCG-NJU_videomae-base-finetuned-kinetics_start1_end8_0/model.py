import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        conv3d = torch.conv3d(in_3, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1);  in_3 = in_1 = in_0 = None
        tmp_4 = conv3d.flatten(2);  conv3d = None
        tmp_5 = tmp_4.transpose(1, 2);  tmp_4 = None
        tmp_6 = in_2.detach();  in_2 = None
        tmp_7 = tmp_6.type_as(tmp_5);  tmp_6 = None
        tmp_8 = tmp_7.to(device = device(type='cuda', index=0), copy = True);  tmp_7 = None
        tmp_9 = tmp_5 + tmp_8;  tmp_5 = tmp_8 = None
        return (tmp_9,)
        