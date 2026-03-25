import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.full((19, 20), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_2 = torch.triu(tmp_1, diagonal=1)
        tmp_1 = None
        tmp_3 = torch.arange(20, device=device(type='cuda', index=0))
        tmp_4 = in_1.reshape(-1, 1)
        tmp_5 = tmp_3 > tmp_4
        tmp_3 = tmp_4 = None
        tmp_2 *= tmp_5
        tmp_6 = tmp_2
        tmp_2 = tmp_5 = None
        tmp_7 = tmp_6[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_6 = None
        tmp_8 = tmp_7.expand(1, 1, -1, -1)
        tmp_7 = None
        tmp_9 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_10 = tmp_9.expand(1, 1, 19, 19)
        tmp_9 = None
        return (tmp_8, tmp_10)