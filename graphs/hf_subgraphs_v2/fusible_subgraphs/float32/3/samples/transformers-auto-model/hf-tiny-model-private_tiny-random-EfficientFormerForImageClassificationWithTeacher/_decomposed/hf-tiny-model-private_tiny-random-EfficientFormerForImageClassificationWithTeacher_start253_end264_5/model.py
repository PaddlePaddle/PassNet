import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = torch.nn.functional.linear(in_3, tmp_2, tmp_1)
        tmp_2 = tmp_1 = None
        tmp_4 = tmp_3.reshape(16, 49, 8, -1)
        tmp_3 = None
        tmp_5 = tmp_4.split([32, 32, 128], dim=3)
        tmp_4 = None
        tmp_6 = tmp_5[0]
        tmp_7 = tmp_5[1]
        tmp_8 = tmp_5[2]
        tmp_5 = None
        tmp_9 = tmp_6.permute(0, 2, 1, 3)
        tmp_6 = None
        tmp_10 = tmp_7.permute(0, 2, 1, 3)
        tmp_7 = None
        tmp_11 = tmp_8.permute(0, 2, 1, 3)
        tmp_8 = None
        tmp_12 = tmp_0.to(device(type='cuda', index=0))
        tmp_0 = None
        tmp_13 = tmp_10.transpose(-2, -1)
        tmp_10 = None
        return (tmp_9, tmp_12, tmp_13, tmp_11)