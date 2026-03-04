import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
        tmp_1 = tmp_0[slice(None, None, None), slice(-5, None, None), slice(None, None, None)]
        tmp_2 = tmp_1.fill_(1)
        tmp_1 = tmp_2 = None
        tmp_3 = tmp_0[slice(None, None, None), slice(None, None, None), slice(-5, None, None)]
        tmp_4 = tmp_3.fill_(1)
        tmp_3 = tmp_4 = None
        tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 96)
        tmp_6 = tmp_5.transpose(2, 3)
        tmp_5 = None
        tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
        tmp_0 = None
        tmp_8 = tmp_7.transpose(2, 3)
        tmp_7 = None
        tmp_9 = tmp_8.reshape(1, 361, 49)
        tmp_8 = None
        tmp_10 = tmp_9.unsqueeze(2)
        tmp_11 = tmp_9.unsqueeze(3)
        tmp_9 = None
        tmp_12 = tmp_10 - tmp_11
        tmp_10 = tmp_11 = None
        return (tmp_12, tmp_6)