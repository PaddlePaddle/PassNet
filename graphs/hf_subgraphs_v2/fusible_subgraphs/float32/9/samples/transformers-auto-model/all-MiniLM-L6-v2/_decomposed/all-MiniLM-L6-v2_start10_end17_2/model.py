import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.ones((1, 512), device=device(type='cuda', index=0))
        tmp_1 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_0 = None
        tmp_2 = tmp_1.expand(1, 1, 512, 512)
        tmp_1 = None
        tmp_3 = tmp_2.to(torch.float32)
        tmp_2 = None
        tmp_4 = 1.0 - tmp_3
        tmp_3 = None
        tmp_5 = tmp_4.to(torch.bool)
        tmp_6 = tmp_4.masked_fill(tmp_5, -3.4028234663852886e+38)
        tmp_4 = tmp_5 = None
        return (tmp_6,)