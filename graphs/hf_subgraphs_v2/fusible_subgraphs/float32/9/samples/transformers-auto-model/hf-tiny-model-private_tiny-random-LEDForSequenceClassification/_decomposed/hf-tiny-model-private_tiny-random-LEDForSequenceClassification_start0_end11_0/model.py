import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = torch.full((22, 22), -3.4028234663852886e+38, device=device(type='cuda', index=0))
        tmp_2 = torch.arange(22, device=device(type='cuda', index=0))
        tmp_3 = tmp_2 + 1
        tmp_4 = tmp_3.view(22, 1)
        tmp_3 = None
        tmp_5 = tmp_2 < tmp_4
        tmp_2 = tmp_4 = None
        tmp_6 = tmp_1.masked_fill_(tmp_5, 0)
        tmp_5 = tmp_6 = None
        tmp_7 = tmp_1.to(torch.float32)
        tmp_1 = None
        tmp_8 = tmp_7[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_7 = None
        tmp_9 = tmp_8.expand(1, 1, 22, 22)
        tmp_8 = None
        tmp_10 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_11 = tmp_10.expand(1, 1, 22, 22)
        tmp_10 = None
        return (tmp_9, tmp_11)