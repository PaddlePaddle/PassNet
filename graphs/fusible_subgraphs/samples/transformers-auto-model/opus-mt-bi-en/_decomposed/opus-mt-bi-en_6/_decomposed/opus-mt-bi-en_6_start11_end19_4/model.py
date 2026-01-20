import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.arange(2, device=device(type='cuda'))
        tmp_1 = in_1.reshape(-1, 1)
        tmp_2 = tmp_0 > tmp_1
        tmp_0 = tmp_1 = None
        in_2 *= tmp_2
        tmp_3 = in_2
        tmp_2 = None
        tmp_4 = tmp_3[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_3 = None
        tmp_5 = tmp_4.expand(1, 1, -1, -1)
        tmp_4 = None
        tmp_6 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_7 = tmp_6.expand(1, 1, 1, 29)
        tmp_6 = None
        return (tmp_5, tmp_7)