import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1.cumsum(-1)
        tmp_2 = tmp_1 - 1
        tmp_1 = None
        tmp_3 = tmp_0.__eq__(0)
        tmp_0 = None
        tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
        tmp_3 = tmp_4 = None
        tmp_5 = tmp_2.unsqueeze(0)
        tmp_2 = None
        tmp_6 = tmp_5.expand(3, -1, -1)
        tmp_5 = None
        tmp_7 = tmp_6.to(device(type='cuda', index=0))
        tmp_6 = None
        tmp_8 = tmp_7.max(0, keepdim=False)
        tmp_9 = tmp_8[0]
        tmp_8 = None
        tmp_10 = tmp_9.max(-1, keepdim=True)
        tmp_9 = None
        tmp_11 = tmp_10[0]
        tmp_10 = None
        tmp_12 = tmp_11 + 1
        tmp_11 = None
        tmp_13 = tmp_12 - 9
        tmp_12 = None
        return (tmp_13, tmp_7)