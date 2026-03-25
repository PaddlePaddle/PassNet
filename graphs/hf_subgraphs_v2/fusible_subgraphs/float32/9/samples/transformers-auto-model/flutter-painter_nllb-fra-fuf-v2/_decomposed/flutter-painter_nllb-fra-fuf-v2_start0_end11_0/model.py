import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = in_0
        tmp_1 = torch.arange(0, 16, device=device(type='cuda', index=0))
        tmp_2 = torch.full((16, 17), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
        tmp_3 = torch.triu(tmp_2, diagonal=1)
        tmp_2 = None
        tmp_4 = torch.arange(17, device=device(type='cuda', index=0))
        tmp_5 = tmp_1.reshape(-1, 1)
        tmp_1 = None
        tmp_6 = tmp_4 > tmp_5
        tmp_4 = tmp_5 = None
        tmp_3 *= tmp_6
        tmp_7 = tmp_3
        tmp_3 = tmp_6 = None
        tmp_8 = tmp_7[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_7 = None
        tmp_9 = tmp_8.expand(1, 1, -1, -1)
        tmp_8 = None
        tmp_10 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_11 = tmp_10.expand(1, 1, 16, 16)
        tmp_10 = None
        return (tmp_9, tmp_11)