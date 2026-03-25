import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.conv3d(in_3, tmp_1, tmp_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
        tmp_1 = tmp_0 = None
        tmp_4 = tmp_3.flatten(2)
        tmp_3 = None
        tmp_5 = tmp_4.transpose(1, 2)
        tmp_4 = None
        tmp_6 = tmp_2.detach()
        tmp_2 = None
        tmp_7 = tmp_6.type_as(tmp_5)
        tmp_6 = None
        tmp_8 = tmp_7.to(device=device(type='cuda', index=0), copy=True)
        tmp_7 = None
        tmp_9 = tmp_5 + tmp_8
        tmp_5 = tmp_8 = None
        return (tmp_9,)